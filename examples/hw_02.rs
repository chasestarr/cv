use cv::image::{bilinear_resize, load_image, nn_resize, save_image};
use std::path::Path;

fn nearest_neighbor() {
  let im = load_image(Path::new("images/dog.jpg")).unwrap();
  let resized = nn_resize(&im, im.width * 4, im.height * 4);
  // let resized = nn_resize(&im, im.width / 7, im.height / 7);
  save_image(&resized, Path::new("tmp/hw_02_nn_resize.jpg")).unwrap();
}

fn bilinear() {
  let im = load_image(Path::new("images/dog.jpg")).unwrap();
  // let resized = bilinear_resize(&im, im.width * 4, im.height * 4);
  let resized = bilinear_resize(&im, im.width / 7, im.height / 7);
  save_image(&resized, Path::new("tmp/hw_02_bilinear_resize.jpg")).unwrap();
}

fn main() {
  nearest_neighbor();
  bilinear();
}
