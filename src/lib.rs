use palette::{Oklab, Srgb, IntoColor};

/// Error types for dithering operations
#[derive(Debug, Clone, PartialEq)]
pub enum DitherError {
    /// Image buffer length is not divisible by 3
    InvalidImageBufferSize,
    /// Palette buffer length is not divisible by 3
    InvalidPaletteBufferSize,
    /// Palette is empty
    EmptyPalette,
    /// width × height × 3 does not match buffer length
    DimensionMismatch { expected: usize, actual: usize },
}

/// 8x8 Bayer threshold matrix, normalized to [-0.5, 0.5]
const BAYER_8X8: [[f32; 8]; 8] = [
    [ 0.0/64.0 - 0.5, 32.0/64.0 - 0.5,  8.0/64.0 - 0.5, 40.0/64.0 - 0.5,  2.0/64.0 - 0.5, 34.0/64.0 - 0.5, 10.0/64.0 - 0.5, 42.0/64.0 - 0.5],
    [48.0/64.0 - 0.5, 16.0/64.0 - 0.5, 56.0/64.0 - 0.5, 24.0/64.0 - 0.5, 50.0/64.0 - 0.5, 18.0/64.0 - 0.5, 58.0/64.0 - 0.5, 26.0/64.0 - 0.5],
    [12.0/64.0 - 0.5, 44.0/64.0 - 0.5,  4.0/64.0 - 0.5, 36.0/64.0 - 0.5, 14.0/64.0 - 0.5, 46.0/64.0 - 0.5,  6.0/64.0 - 0.5, 38.0/64.0 - 0.5],
    [60.0/64.0 - 0.5, 28.0/64.0 - 0.5, 52.0/64.0 - 0.5, 20.0/64.0 - 0.5, 62.0/64.0 - 0.5, 30.0/64.0 - 0.5, 54.0/64.0 - 0.5, 22.0/64.0 - 0.5],
    [ 3.0/64.0 - 0.5, 35.0/64.0 - 0.5, 11.0/64.0 - 0.5, 43.0/64.0 - 0.5,  1.0/64.0 - 0.5, 33.0/64.0 - 0.5,  9.0/64.0 - 0.5, 41.0/64.0 - 0.5],
    [51.0/64.0 - 0.5, 19.0/64.0 - 0.5, 59.0/64.0 - 0.5, 27.0/64.0 - 0.5, 49.0/64.0 - 0.5, 17.0/64.0 - 0.5, 57.0/64.0 - 0.5, 25.0/64.0 - 0.5],
    [15.0/64.0 - 0.5, 47.0/64.0 - 0.5,  7.0/64.0 - 0.5, 39.0/64.0 - 0.5, 13.0/64.0 - 0.5, 45.0/64.0 - 0.5,  5.0/64.0 - 0.5, 37.0/64.0 - 0.5],
    [63.0/64.0 - 0.5, 31.0/64.0 - 0.5, 55.0/64.0 - 0.5, 23.0/64.0 - 0.5, 61.0/64.0 - 0.5, 29.0/64.0 - 0.5, 53.0/64.0 - 0.5, 21.0/64.0 - 0.5],
];

/// Get Bayer threshold for pixel position with modulo wrapping
fn bayer_threshold(x: u32, y: u32) -> f32 {
    let x_mod = (x % 8) as usize;
    let y_mod = (y % 8) as usize;
    BAYER_8X8[y_mod][x_mod]
}

/// Convert RGB u8 triplet to Oklab using palette crate
fn rgb_to_oklab(r: u8, g: u8, b: u8) -> Oklab<f32> {
    let srgb = Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    srgb.into_color()
}

/// Convert Oklab back to RGB u8 triplet
#[allow(dead_code)] // Used in tests
fn oklab_to_rgb(oklab: Oklab<f32>) -> (u8, u8, u8) {
    let srgb: Srgb<f32> = oklab.into_color();
    (
        (srgb.red * 255.0).round() as u8,
        (srgb.green * 255.0).round() as u8,
        (srgb.blue * 255.0).round() as u8,
    )
}

/// Returns squared Oklab distance (skip sqrt for comparison efficiency)
fn oklab_distance_squared(a: Oklab<f32>, b: Oklab<f32>) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    dl * dl + da * da + db * db
}

/// Find nearest palette color in Oklab space
fn find_nearest_palette_color(
    target: Oklab<f32>,
    palette_oklab: &[Oklab<f32>],
    palette_rgb: &[(u8, u8, u8)],
) -> (u8, u8, u8) {
    debug_assert_eq!(palette_oklab.len(), palette_rgb.len());
    debug_assert!(!palette_oklab.is_empty());

    let mut min_distance = f32::INFINITY;
    let mut best_color = palette_rgb[0];

    for (oklab_color, rgb_color) in palette_oklab.iter().zip(palette_rgb.iter()) {
        let distance = oklab_distance_squared(target, *oklab_color);
        if distance < min_distance {
            min_distance = distance;
            best_color = *rgb_color;
        }
    }

    best_color
}

/// Scaling factor for dither strength in Oklab L channel.
/// Oklab L ranges from 0 to 1, so 0.12 gives good visible dithering.
const L_SCALE: f32 = 0.12;

/// Applies Bayer 8x8 ordered dithering to an image in Oklab color space.
///
/// # Arguments
/// * `image` - Input image as RGB8 pixels
/// * `palette` - Color palette as RGB8 values (must not be empty)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `dither_strength` - Dithering intensity (0.0 = none, 1.0 = full)
///
/// # Returns
/// * `Ok(Vec<u8>)` - Dithered image as RGB8 pixels
/// * `Err(DitherError)` - If inputs are invalid
pub fn dither_bayer_oklab(
    image: &[u8],
    palette: &[u8],
    width: u32,
    height: u32,
    dither_strength: f32,
) -> Result<Vec<u8>, DitherError> {
    // input validation
    if image.len() % 3 != 0 {
        return Err(DitherError::InvalidImageBufferSize);
    }
    if palette.len() % 3 != 0 {
        return Err(DitherError::InvalidPaletteBufferSize);
    }
    if palette.is_empty() {
        return Err(DitherError::EmptyPalette);
    }
    let expected_len = (width as usize) * (height as usize) * 3;
    if image.len() != expected_len {
        return Err(DitherError::DimensionMismatch {
            expected: expected_len,
            actual: image.len(),
        });
    }

    // pre compute OKlab palette
    let palette_rgb: Vec<(u8, u8, u8)> = palette
        .chunks_exact(3)
        .map(|rgb| (rgb[0], rgb[1], rgb[2]))
        .collect();
    let palette_oklab: Vec<Oklab<f32>> = palette_rgb
        .iter()
        .map(|&(r, g, b)| rgb_to_oklab(r, g, b))
        .collect();

    let mut output = vec![0u8; image.len()];

    // process pixel by pixel
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y as usize) * (width as usize) + (x as usize)) * 3;
            let r = image[pixel_idx];
            let g = image[pixel_idx + 1];
            let b = image[pixel_idx + 2];

            let mut oklab = rgb_to_oklab(r, g, b);

            let threshold = bayer_threshold(x, y);
            oklab.l = (oklab.l + threshold * dither_strength * L_SCALE).clamp(0.0, 1.0);

            let (pr, pg, pb) = find_nearest_palette_color(oklab, &palette_oklab, &palette_rgb);

            output[pixel_idx] = pr;
            output[pixel_idx + 1] = pg;
            output[pixel_idx + 2] = pb;
        }
    }

    Ok(output)
}

/// Zero alloc variant that writes to a pre allocated output buffer
/// Same logic as dither_bayer_oklab but avoids allocation
pub fn dither_bayer_oklab_into(
    image: &[u8],
    palette: &[u8],
    width: u32,
    height: u32,
    dither_strength: f32,
    output: &mut [u8], // pre allocated mutable output buffer
) -> Result<(), DitherError> {
    // input validation (same as dither_bayer_oklab)
    if image.len() % 3 != 0 {
        return Err(DitherError::InvalidImageBufferSize);
    }
    if palette.len() % 3 != 0 {
        return Err(DitherError::InvalidPaletteBufferSize);
    }
    if palette.is_empty() {
        return Err(DitherError::EmptyPalette);
    }
    let expected_len = (width as usize) * (height as usize) * 3;
    if image.len() != expected_len {
        return Err(DitherError::DimensionMismatch {
            expected: expected_len,
            actual: image.len(),
        });
    }
    if output.len() != expected_len {
        return Err(DitherError::DimensionMismatch {
            expected: expected_len,
            actual: output.len(),
        });
    }

    let palette_rgb: Vec<(u8, u8, u8)> = palette
        .chunks_exact(3)
        .map(|rgb| (rgb[0], rgb[1], rgb[2]))
        .collect();
    let palette_oklab: Vec<Oklab<f32>> = palette_rgb
        .iter()
        .map(|&(r, g, b)| rgb_to_oklab(r, g, b))
        .collect();

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y as usize) * (width as usize) + (x as usize)) * 3;
            let r = image[pixel_idx];
            let g = image[pixel_idx + 1];
            let b = image[pixel_idx + 2];

            let mut oklab = rgb_to_oklab(r, g, b);

            let threshold = bayer_threshold(x, y);
            oklab.l = (oklab.l + threshold * dither_strength * L_SCALE).clamp(0.0, 1.0);

            let (pr, pg, pb) = find_nearest_palette_color(oklab, &palette_oklab, &palette_rgb);

            output[pixel_idx] = pr;
            output[pixel_idx + 1] = pg;
            output[pixel_idx + 2] = pb;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    // property based tests
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// PROPERTY: RGB/OKlab round trip consistency
        /// For any valid sRGB color (r, g, b) where each component is in [0, 255],
        /// converting to Oklab and back to sRGB shall produce a color where each component
        /// differs by at most 2 from the original (accounting for floating-point and gamma correction rounding)
        #[test]
        fn rgb_oklab_round_trip_consistency(r: u8, g: u8, b: u8) {
            let oklab = rgb_to_oklab(r, g, b);
            let (r2, g2, b2) = oklab_to_rgb(oklab);

            // allow for small rounding errors due to floating point and gamma correction
            prop_assert!((r2 as i16 - r as i16).abs() <= 2);
            prop_assert!((g2 as i16 - g as i16).abs() <= 2);
            prop_assert!((b2 as i16 - b as i16).abs() <= 2);
        }

        /// PROPERTY: Palette matching returns minimum distance
        /// For any target Oklab color and non empty palette, the returned palette color
        /// shall have euclidean distance less than or equal to all other palette colors' distances to the target
        #[test]
        fn palette_matching_returns_minimum_distance(
            target_r: u8, target_g: u8, target_b: u8,
            palette in prop::collection::vec((0u8..=255, 0u8..=255, 0u8..=255), 1..10)
        ) {
            let target_oklab = rgb_to_oklab(target_r, target_g, target_b);
            let palette_oklab: Vec<Oklab<f32>> = palette.iter()
                .map(|&(r, g, b)| rgb_to_oklab(r, g, b))
                .collect();

            let result_rgb = find_nearest_palette_color(target_oklab, &palette_oklab, &palette);
            let result_oklab = rgb_to_oklab(result_rgb.0, result_rgb.1, result_rgb.2);
            let min_distance = oklab_distance_squared(target_oklab, result_oklab);

            // verify this is indeed the minimum distance
            for &palette_color in &palette {
                let palette_oklab = rgb_to_oklab(palette_color.0, palette_color.1, palette_color.2);
                let distance = oklab_distance_squared(target_oklab, palette_oklab);
                prop_assert!(min_distance <= distance);
            }
        }

        /// PROPERTY: Bayer threshold wraps at 8x8
        /// For any pixel coordinates (x, y), bayer_threshold(x, y) shall equal bayer_threshold(x % 8, y % 8)
        #[test]
        fn bayer_threshold_wraps_at_8x8(x: u32, y: u32) {
            let threshold = bayer_threshold(x, y);
            let wrapped_threshold = bayer_threshold(x % 8, y % 8);
            prop_assert_eq!(threshold, wrapped_threshold);
        }

        /// PROPERTY: Output buffer size invariant
        /// For any valid image buffer of length n, the dithered output buffer shall also have length n
        #[test]
        fn output_buffer_size_invariant(
            width in 1u32..20,
            height in 1u32..20,
            image in prop::collection::vec(0u8..=255, 3..300).prop_filter("Must be divisible by 3", |v| v.len() % 3 == 0),
            palette in prop::collection::vec(0u8..=255, 3..30).prop_filter("Must be divisible by 3", |v| v.len() % 3 == 0 && !v.is_empty()),
            dither_strength in 0.0f32..1.0f32,
        ) {
            // only test when image size matches dimensions
            let expected_size = (width as usize) * (height as usize) * 3;
            if image.len() == expected_size {
                let result = dither_bayer_oklab(&image, &palette, width, height, dither_strength);
                if let Ok(output) = result {
                    prop_assert_eq!(output.len(), image.len());
                }
            }
        }
    }

    // unit tests for specific examples and edge cases

    /// PROPERTY: Output pixels are palette members
    /// For any valid image and palette, every pixel in the output buffer shall exactly match one of the palette colors
    #[test]
    fn output_pixels_are_palette_members() {
        let width = 2u32;
        let height = 1u32;
        let image = vec![255, 0, 0, 0, 255, 0]; // 2 pixels
        let palette = vec![255, 0, 0, 0, 255, 0]; // 2 colors

        let output = dither_bayer_oklab(&image, &palette, width, height, 0.5).unwrap();
        let palette_colors: HashSet<(u8, u8, u8)> = palette
            .chunks_exact(3)
            .map(|rgb| (rgb[0], rgb[1], rgb[2]))
            .collect();

        for pixel in output.chunks_exact(3) {
            let rgb = (pixel[0], pixel[1], pixel[2]);
            assert!(
                palette_colors.contains(&rgb),
                "Output pixel {:?} is not in palette {:?}",
                rgb,
                palette_colors
            );
        }
    }

    /// PROPERTY: Dithering is deterministic
    /// For any valid inputs, calling dither_bayer_oklab twice with identical arguments
    /// shall produce identical output buffers
    #[test]
    fn dithering_is_deterministic() {
        let width = 2u32;
        let height = 1u32;
        let image = vec![255, 128, 64, 32, 16, 8]; // 2 pixels
        let palette = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 colors

        let output1 = dither_bayer_oklab(&image, &palette, width, height, 0.5).unwrap();
        let output2 = dither_bayer_oklab(&image, &palette, width, height, 0.5).unwrap();
        assert_eq!(output1, output2);
    }

    /// PROPERTY: Zero strength equals nearest neighbor
    /// For any valid image and palette, when dither_strength is 0.0, the output shall be identical
    /// to simple nearest neighbor quantization in Oklab space (no position dependent variation)
    #[test]
    fn zero_strength_equals_nearest_neighbor() {
        let width = 2u32;
        let height = 1u32;
        let image = vec![255, 128, 64, 32, 16, 8]; // 2 pixels
        let palette = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 colors

        let dithered_output = dither_bayer_oklab(&image, &palette, width, height, 0.0).unwrap();

        // manually compute nearest neighbor quantization
        let palette_rgb: Vec<(u8, u8, u8)> = palette
            .chunks_exact(3)
            .map(|rgb| (rgb[0], rgb[1], rgb[2]))
            .collect();
        let palette_oklab: Vec<Oklab<f32>> = palette_rgb
            .iter()
            .map(|&(r, g, b)| rgb_to_oklab(r, g, b))
            .collect();

        let mut expected_output = vec![0u8; image.len()];
        for (i, pixel) in image.chunks_exact(3).enumerate() {
            let oklab = rgb_to_oklab(pixel[0], pixel[1], pixel[2]);
            let (r, g, b) = find_nearest_palette_color(oklab, &palette_oklab, &palette_rgb);
            let idx = i * 3;
            expected_output[idx] = r;
            expected_output[idx + 1] = g;
            expected_output[idx + 2] = b;
        }

        assert_eq!(dithered_output, expected_output);
    }
 
    fn hex_str_to_u8(hex_str: &str) -> u8 {
        u8::from_str_radix(hex_str, 16).unwrap()
    }
    
    fn parse_hex_color(hex: &str) -> (u8, u8, u8) {
        if hex.len() != 6 {
            panic!("Invalid hex color: {}, expected 6 characters", hex);
        }
        let r_hex = &hex[0..2];
        let g_hex = &hex[2..4];
        let b_hex = &hex[4..6];
        let r = hex_str_to_u8(r_hex);
        let g = hex_str_to_u8(g_hex);
        let b = hex_str_to_u8(b_hex);
        (r, g, b)
    }
    
    fn palette_from_string(palette_string: &str) -> Vec<u8> {
        let mut palette: Vec<u8> = Vec::new();
        for line in palette_string.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (r, g, b) = parse_hex_color(line);
            palette.push(r);
            palette.push(g);
            palette.push(b);
        }
        palette
    } 

    // error conditions

    #[test]
    fn test_invalid_image_buffer_size() {
        let image = vec![255, 0]; // Not divisible by 3
        let palette = vec![255, 0, 0];
        let result = dither_bayer_oklab(&image, &palette, 1, 1, 0.5);
        assert_eq!(result, Err(DitherError::InvalidImageBufferSize));
    }

    #[test]
    fn test_invalid_palette_buffer_size() {
        let image = vec![255, 0, 0];
        let palette = vec![255, 0]; // Not divisible by 3
        let result = dither_bayer_oklab(&image, &palette, 1, 1, 0.5);
        assert_eq!(result, Err(DitherError::InvalidPaletteBufferSize));
    }

    #[test]
    fn test_empty_palette() {
        let image = vec![255, 0, 0];
        let palette = vec![]; // Empty palette
        let result = dither_bayer_oklab(&image, &palette, 1, 1, 0.5);
        assert_eq!(result, Err(DitherError::EmptyPalette));
    }

    #[test]
    fn test_dimension_mismatch() {
        let image = vec![255, 0, 0]; // 1 pixel
        let palette = vec![255, 0, 0];
        let result = dither_bayer_oklab(&image, &palette, 2, 2, 0.5); // Claims to be 2x2 = 4 pixels
        assert_eq!(
            result,
            Err(DitherError::DimensionMismatch {
                expected: 12,
                actual: 3
            })
        );
    }

    #[test]
    fn test_bayer_matrix_values() {
        // test specific known values from the Bayer matrix
        let threshold_0_0 = bayer_threshold(0, 0);
        assert_eq!(threshold_0_0, 0.0 / 64.0 - 0.5);

        let threshold_7_7 = bayer_threshold(7, 7);
        assert_eq!(threshold_7_7, 21.0 / 64.0 - 0.5);

        // test wrapping at 8
        let threshold_8_8 = bayer_threshold(8, 8);
        assert_eq!(threshold_8_8, threshold_0_0);
    }

    /// REGRESSION: SIMD optimizations produce byte identical output (if i decided to add them)
    /// 
    /// Validates that algorithm-level optimizations don't change the 
    /// dithering output due to operation reordering or precision differences
    #[test]
    fn test_regression_known_output() {
        let palette_data = include_str!("../palettes/palette1.hex");
        let palette = palette_from_string(palette_data);
    
        let image_data = include_bytes!("../images/dog_small.png");
        let image = image::load_from_memory(image_data).unwrap();
    
        let rgb_image = image.to_rgb8();
        let image_bytes = rgb_image.as_raw();
        let width = rgb_image.width();
        let height = rgb_image.height();
    
        // Apply dithering
        let dither_strength = 0.5;
        let dithered_bytes = dither_bayer_oklab(
            image_bytes,
            &palette,
            width,
            height,
            dither_strength,
        ).expect("Dithering failed");
    
        let ground_truth_dithered_bytes = include_bytes!("../images/dithered_dog_small.png");
        let ground_truth_dithered_image = image::load_from_memory(ground_truth_dithered_bytes).unwrap();
        let ground_truth_dithered_rgb_image = ground_truth_dithered_image.to_rgb8();
        let ground_truth_dithered_image_bytes = ground_truth_dithered_rgb_image.as_raw().clone();
    
        assert_eq!(dithered_bytes, ground_truth_dithered_image_bytes);
    }
}
