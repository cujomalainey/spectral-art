extern crate ini;

use ini::Ini;
use std::env;
use wavefile::WaveFile;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

const SECTION_MUSIC: &str = "music";
const MUSIC_FILE: &str = "file";
const MUSIC_START_TIME: &str = "start_time";
const MUSIC_STOP_TIME: &str = "stop_time";

const SECTION_IMAGE: &str = "image";
const IMAGE_WIDTH: &str = "width";
const IMAGE_HEIGHT: &str = "height";
const IMAGE_GRADIENT_FILE: &str = "gradient_file";
const IMAGE_OUTPUT_FILE: &str = "output_file";

const SECTION_FFT: &str = "fft";
const FFT_WIDTH: &str = "width";
const FFT_WINDOW_FUNCTION: &str = "window_function";
const FFT_DECIMATIONS: &str = "decimations";

const SECTION_SCALING: &str = "scaling";
const SCALING_FREQUENCY: &str = "frequency";
const SCALING_AMPLITUDE: &str = "amplitude";
const SCALING_SMOOTHING_FACTOR: &str = "smoothing_factor";
const SCALING_LOWER_FREQUENCY: &str = "lower_frequency";
const SCALING_UPPER_FREQUENCY: &str = "upper_frequency";

fn avg(v: &Vec<i32>) -> i32 {
    let x: i32 = v.iter().sum();
    let y: i32 = v.len() as i32;
    x / y
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let conf = Ini::load_from_file(&args[1]).unwrap();

    let audio_section = conf.section(Some(SECTION_MUSIC)).unwrap();
    let file = audio_section.get(MUSIC_FILE).unwrap();
    let start_time = audio_section.get(MUSIC_START_TIME).unwrap();
    let stop_time = audio_section.get(MUSIC_STOP_TIME).unwrap();

    let f = WaveFile::open(file.as_str()).unwrap();
    println!("{:}", f.sample_rate());
    let mut iter = f.iter();

    let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 0];
    let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1024];

    for _i in 0..1024 {
        let frame = iter.nth(0).unwrap();
        input.push(Complex::new(avg(&frame) as f32, 0.0));
    }

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(1024);
    fft.process(&mut input, &mut output);

    // for i in 0..513 {
    for i in 0..4 {
        println!("{:?}", output[i]);
    }
}
