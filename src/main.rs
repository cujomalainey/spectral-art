extern crate ini;

use ini::Ini;
use std::env;
use wavefile::WaveFile;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

fn avg(v: &Vec<i32>) -> i32 {
    let x: i32 = v.iter().sum();
    let y: i32 = v.len() as i32;
    x / y
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let conf = Ini::load_from_file(&args[1]).unwrap();

    let audio_section = conf.section(Some("music")).unwrap();
    let file = audio_section.get("file").unwrap();
    // let start_time = audio_section.get("start_time").unwrap();
    // let stop_time = audio_section.get("start_time").unwrap();

    let f = WaveFile::open(file.as_str()).unwrap();
    println!("{:}", f.sample_rate());
    let mut iter = f.iter();
    // Perform a forward FFT of size 1234

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
