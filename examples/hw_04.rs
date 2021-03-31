use cv::draw::{draw_line, mark};
use cv::harris::corner_detector;
use cv::image::{copy_image, load_image, save_image};
use cv::panorama::{find_and_draw_matches, panorama_image};
use cv::point::Point;
use cv::signature::{normalized_distance, sample_size, signature};
use std::path::Path;

fn detect_and_draw_corners() {
    let im = load_image(Path::new("images/rainier_01.png")).unwrap();

    let descriptors = corner_detector(&im, 2.0, 50.0, 3);

    let mut r = copy_image(&im);
    for i in 0..descriptors.len() {
        mark(&mut r, &descriptors[i].p);
    }

    save_image(&r, Path::new("tmp/hw_04_harris.jpg")).unwrap();
}

fn match_corners() {
    let a = load_image(Path::new("images/rainier_01.png")).unwrap();
    let b = load_image(Path::new("images/rainier_02.png")).unwrap();
    let m = find_and_draw_matches(&a, &b, 2.0, 50.0, 3);
    save_image(&m, Path::new("tmp/hw_04_matches.jpg")).unwrap();
}

fn line() {
    let mut im = load_image(Path::new("images/rainier_01.png")).unwrap();
    draw_line(
        &mut im,
        &Point { x: 200.0, y: 200.0 },
        &Point { x: 300.0, y: 100.0 },
    );
    save_image(&im, Path::new("tmp/hw_04_line.png")).unwrap();
}

fn panorama() {
    let a = load_image(Path::new("images/rainier_01.png")).unwrap();
    let b = load_image(Path::new("images/rainier_02.png")).unwrap();
    let im = panorama_image(&a, &b, 2.0, 3.0, 3, 3.0, 1000, 50).unwrap();
    save_image(&im, Path::new("tmp/hw_04_panorama.png")).unwrap();
}

fn image_hash() {
    let a = load_image(Path::new("images/dog.jpg")).unwrap();
    let b = load_image(Path::new("images/dog_block.png")).unwrap();
    let c = load_image(Path::new("images/rainier_01.png")).unwrap();
    let sig_a = signature(&a, (0.5, 0.95), 9, sample_size(&a), 2.0 / 255.0);
    let sig_b = signature(&b, (0.5, 0.95), 9, sample_size(&b), 2.0 / 255.0);
    let sig_c = signature(&b, (0.5, 0.95), 9, sample_size(&c), 2.0 / 255.0);
    let dist_dog = normalized_distance(&sig_a, &sig_b);
    println!("{:?}", dist_dog);
    let dist_rainier = normalized_distance(&sig_a, &sig_c);
    println!("{:?}", dist_rainier);
}

fn main() {
    detect_and_draw_corners();
    match_corners();
    line();
    // panorama();
    image_hash();
}
