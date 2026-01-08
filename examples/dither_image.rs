use okbayer::dither_bayer_oklab;

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
    // Parse each hex color into RGB bytes
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

fn main() {
    let image_data = include_bytes!("../images/flower.png");
    let image = image::load_from_memory(image_data).unwrap();
    let rgb_image = image.to_rgb8();
    let image_bytes = rgb_image.as_raw();
    let width = rgb_image.width();
    let height = rgb_image.height();

    let palette = include_str!("../palettes/island-joy-16.hex");
    let palette_bytes = palette_from_string(palette);

    let dithered_bytes = dither_bayer_oklab(image_bytes, &palette_bytes, width, height, 0.0).unwrap();
    let dithered_image = image::RgbImage::from_raw(width, height, dithered_bytes).unwrap();
    dithered_image.save("images/dithered_flower_0.0.png").unwrap();
}