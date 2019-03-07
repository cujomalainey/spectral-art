extern crate ini;
extern crate image;

use ini::Ini;
use image::{ImageBuffer, Rgb};
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::num_traits::One;
use std::env;
use std::collections::HashMap;
use wavefile::WaveFile;

const SECTION_MUSIC: &str = "music";
const MUSIC_FILE: &str = "file";
const MUSIC_IS_MONO: &str = "is_mono";
const MUSIC_START_TIME: &str = "start_time";
const MUSIC_STOP_TIME: &str = "stop_time";
const MUSIC_CHANNEL: &str = "channel";

const SECTION_IMAGE: &str = "image";
const IMAGE_WIDTH: &str = "width";
const IMAGE_HEIGHT: &str = "height";
const IMAGE_GRADIENT_FILE: &str = "gradient_file";
const IMAGE_OUTPUT_FILE: &str = "output_file";

const SECTION_FFT: &str = "fft";
const FFT_WIDTH: &str = "width";
const FFT_WINDOW_FUNCTION: &str = "window_function";
const FFT_DECIMATIONS: &str = "decimations";
const FFT_INTERPOLATION: &str = "interpolation";

const SECTION_SCALING: &str = "scaling";
const SCALING_FREQUENCY: &str = "frequency";
const SCALING_AMPLITUDE: &str = "amplitude";
const SCALING_SMOOTHING_FACTOR: &str = "smoothing_factor";
const SCALING_LOWER_FREQUENCY: &str = "lower_frequency";
const SCALING_UPPER_FREQUENCY: &str = "upper_frequency";

type WindowFunctionType = fn(usize) -> Vec<f32>;

fn avg(v: &Vec<i32>) -> i32 {
    let x: i32 = v.iter().sum();
    let y: i32 = v.len() as i32;
    x / y
}

fn load_audio_file(audio_section: &HashMap<String, String>) -> WaveFile {
    let file = audio_section.get(MUSIC_FILE).unwrap();
    let channels = audio_section.get(MUSIC_CHANNEL).unwrap();
    let start_time: usize = audio_section.get(MUSIC_START_TIME).unwrap().parse().unwrap();
    let stop_time: usize = audio_section.get(MUSIC_STOP_TIME).unwrap().parse().unwrap();
    let f = WaveFile::open(file.as_str()).unwrap();
    let runtime = f.len()/f.sample_rate();

    // Dump summary
    println!("Loading audio file: {}", file);
    println!("  Sample rate:  {}", f.sample_rate());
    println!("  Sample width: {}", f.bits_per_sample());
    println!("  Channels:     {}", f.channels());
    println!("  Length:       {}s", runtime);

    // check start and end are within range
    if start_time > runtime || stop_time > runtime {
        panic!("Config time bounds longer than file");
    }
    // check channel config is valid
    if channels != "mono" {
        let channel_count: usize = channels.parse().unwrap();
        if channel_count >= f.channels() {
            panic!("Channel index out of range");
        }
    }
    f
}

fn load_gradient_file(image_section: &HashMap<String, String>) -> Vec<Rgb<u8>> {
    let mut scale: Vec<Rgb<u8>> = Vec::new();
    let file = image_section.get(IMAGE_GRADIENT_FILE).unwrap();
    // TODO load gradient file
    scale
}

fn create_image(image_section: &HashMap<String, String>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let img_width: u32 = image_section.get(IMAGE_WIDTH).unwrap().parse().unwrap();
    let img_height: u32 = image_section.get(IMAGE_HEIGHT).unwrap().parse().unwrap();
    let mut imgbuf = ImageBuffer::new(img_width, img_height);
    imgbuf
}

// TODO implement long list of functions
fn window_rectangle(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement matching to functions
fn get_window_function(fft_section: &HashMap<String, String>) -> WindowFunctionType {
    let window_function = fft_section.get(FFT_WINDOW_FUNCTION).unwrap();
    window_rectangle
}

fn get_fft_window(fft_section: &HashMap<String, String>) -> Vec<f32> {
    let fft_width: usize = fft_section.get(FFT_WIDTH).unwrap().parse().unwrap();
    get_window_function(fft_section)(fft_width)
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let conf = Ini::load_from_file(&args[1]).unwrap();

    let audio_section = conf.section(Some(SECTION_MUSIC)).unwrap();
    let image_section = conf.section(Some(SECTION_IMAGE)).unwrap();
    let fft_section = conf.section(Some(SECTION_FFT)).unwrap();
    let fft_decimations: u8 = fft_section.get(FFT_DECIMATIONS).unwrap().parse().unwrap();
    // let channels:   u8 = audio_section.get(MUSIC_CHANNEL).unwrap().parse().unwrap();
    // let is_mono:    bool = audio_section.get(MUSIC_IS_MONO).unwrap().parse().unwrap();
    // let start_time: f32 = audio_section.get(MUSIC_START_TIME).unwrap().parse().unwrap();
    // let stop_time:  f32 = audio_section.get(MUSIC_STOP_TIME).unwrap().parse().unwrap();

    let f = load_audio_file(audio_section);
    let img = create_image(image_section);
    let gradient = load_gradient_file(image_section);
    let window = get_fft_window(fft_section);

    let mut iter = f.iter();

    // let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 0];
    // let mut output: Vec<Complex<f32>> = vec![Zero::zero(); fft_width];

    // for _i in 0..fft_width as i32 {
    //     let frame = iter.nth(0).unwrap();
    //     input.push(Complex::new(avg(&frame) as f32, 0.0));
    // }
    //
    // let mut planner = FFTplanner::new(false);
    // let fft = planner.plan_fft(fft_width);
    // fft.process(&mut input, &mut output);
    //
    // // for i in 0..513 {
    // for i in 0..4 {
    //     println!("{:?}", output[i]);
    // }
}
