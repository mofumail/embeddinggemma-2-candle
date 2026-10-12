//! Image preprocessing: aspect-ratio preserving resize to a patch budget, then patchification.

use candle::{Device, Result, Tensor};
use image::imageops::FilterType;
use image::DynamicImage;
use serde::Deserialize;

use crate::vision::PADDING_POSITION;

fn default_patch_size() -> usize {
    16
}
fn default_pooling_kernel_size() -> usize {
    3
}
fn default_image_max_soft_tokens() -> usize {
    280
}

/// Mirrors the `image_processor` and `video_processor` entries of `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct ImageProcessor {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    /// Upper bound on the soft tokens an image is turned into; its resolution is chosen to
    /// use as much of that budget as its aspect ratio allows.
    #[serde(default = "default_image_max_soft_tokens")]
    pub max_soft_tokens: usize,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            patch_size: default_patch_size(),
            pooling_kernel_size: default_pooling_kernel_size(),
            max_soft_tokens: default_image_max_soft_tokens(),
        }
    }
}

/// A resized and patchified image.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePatches {
    /// Row-major patches, each `patch_size * patch_size * 3` values in `[0, 1]`, laid out as
    /// (row, column, channel) within the patch.
    pub pixel_values: Vec<f32>,
    /// Patches per column and per row of the image.
    pub grid_height: usize,
    pub grid_width: usize,
    /// Number of soft tokens the vision tower pools these patches into.
    pub num_soft_tokens: usize,
}

impl ImagePatches {
    pub fn num_patches(&self) -> usize {
        self.grid_height * self.grid_width
    }
}

impl ImageProcessor {
    fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// The `(height, width)` an image is resized to: the largest size of the same aspect
    /// ratio that fits in the patch budget with both sides a multiple of
    /// `pooling_kernel_size * patch_size`.
    pub fn target_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        if height == 0 || width == 0 {
            candle::bail!("cannot process an empty image ({width}x{height})")
        }
        let max_patches = self.max_patches();
        let target_px = max_patches * self.patch_size * self.patch_size;
        let factor = (target_px as f64 / (height * width) as f64).sqrt();
        let side_mult = self.pooling_kernel_size * self.patch_size;
        let round_down =
            |side: usize| (factor * side as f64 / side_mult as f64).floor() as usize * side_mult;
        let (mut target_height, mut target_width) = (round_down(height), round_down(width));

        // Very elongated images round to zero along their short side.
        let max_side_length =
            max_patches / (self.pooling_kernel_size * self.pooling_kernel_size) * side_mult;
        match (target_height, target_width) {
            (0, 0) => {
                candle::bail!("a {width}x{height} image cannot be resized to the patch budget")
            }
            (0, _) => {
                target_height = side_mult;
                target_width = ((width / height) * side_mult).min(max_side_length);
            }
            (_, 0) => {
                target_width = side_mult;
                target_height = ((height / width) * side_mult).min(max_side_length);
            }
            _ => (),
        }
        if target_height * target_width > target_px {
            candle::bail!(
                "resizing {width}x{height} to {target_width}x{target_height} exceeds {max_patches} patches"
            )
        }
        Ok((target_height, target_width))
    }

    /// Resizes (bicubic, antialiased) and patchifies an image.
    pub fn preprocess(&self, image: &DynamicImage) -> Result<ImagePatches> {
        let image = image.to_rgb8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let (target_height, target_width) = self.target_size(height, width)?;
        let image = if (target_height, target_width) == (height, width) {
            image
        } else {
            image::imageops::resize(
                &image,
                target_width as u32,
                target_height as u32,
                FilterType::CatmullRom,
            )
        };

        let ps = self.patch_size;
        let (grid_height, grid_width) = (target_height / ps, target_width / ps);
        let pixels = image.as_raw();
        let mut pixel_values = Vec::with_capacity(grid_height * grid_width * ps * ps * 3);
        for patch_y in 0..grid_height {
            for patch_x in 0..grid_width {
                for y in 0..ps {
                    let start = ((patch_y * ps + y) * target_width + patch_x * ps) * 3;
                    let row = &pixels[start..start + ps * 3];
                    pixel_values.extend(row.iter().map(|&value| f32::from(value) / 255.));
                }
            }
        }
        let k_squared = self.pooling_kernel_size * self.pooling_kernel_size;
        Ok(ImagePatches {
            pixel_values,
            grid_height,
            grid_width,
            num_soft_tokens: grid_height * grid_width / k_squared,
        })
    }

    /// Stacks images into the inputs of [`crate::vision::VisionTower::forward`], padding them
    /// to the patch count of the largest one: `(batch, num_patches, patch_dim)` pixel values
    /// and `(batch, num_patches, 2)` patch positions.
    pub fn collate(&self, images: &[&ImagePatches], device: &Device) -> Result<(Tensor, Tensor)> {
        let patch_dim = self.patch_size * self.patch_size * 3;
        let num_patches = images
            .iter()
            .map(|image| image.num_patches())
            .max()
            .unwrap_or(0);
        let mut pixel_values = Vec::with_capacity(images.len() * num_patches * patch_dim);
        let mut positions = Vec::with_capacity(images.len() * num_patches * 2);
        for image in images {
            pixel_values.extend_from_slice(&image.pixel_values);
            for y in 0..image.grid_height {
                for x in 0..image.grid_width {
                    positions.extend([x as i64, y as i64]);
                }
            }
            let padding = num_patches - image.num_patches();
            pixel_values.resize(pixel_values.len() + padding * patch_dim, 0.);
            positions.resize(positions.len() + padding * 2, PADDING_POSITION);
        }
        let shape = (images.len(), num_patches);
        Ok((
            Tensor::from_vec(pixel_values, (shape.0, shape.1, patch_dim), device)?,
            Tensor::from_vec(positions, (shape.0, shape.1, 2), device)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes_match_the_reference() {
        let processor = ImageProcessor::default();
        // (height, width) -> (height, width), from `get_aspect_ratio_preserving_size`.
        assert_eq!(processor.target_size(480, 640).unwrap(), (672, 912));
        assert_eq!(processor.target_size(768, 768).unwrap(), (768, 768));
        assert_eq!(processor.target_size(1000, 300).unwrap(), (1440, 432));
        assert_eq!(processor.target_size(10, 4000).unwrap(), (48, 13440));
        let video = ImageProcessor {
            max_soft_tokens: 140,
            ..processor
        };
        assert_eq!(video.target_size(480, 640).unwrap(), (480, 624));
    }
}
