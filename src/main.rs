extern crate ini;
extern crate image;

use ini::Ini;
use image::{ImageBuffer, Rgb};
use rustfft::{FFTplanner, FFT};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero, One};
use std::env;
use std::cmp::max;
use std::sync::Arc;
use std::collections::HashMap;
use wavefile::{WaveFile, WaveFileIterator};

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

#[derive(Clone, Copy)]
enum InterpFunc {
    Linear,
    Quadratic,
    Cubic,
}

struct AudioIndex {
    start_frame: usize,
    stop_frame: usize,
    step_size: usize,
    current_step: usize,
}

struct FFTBuilder {
    fft: Arc<dyn FFT<f32>>,
    builder: SampleBuilder,
    decimations: u8,
    buffer_size: usize,
    fft_width: usize,
    sampling_frequency: usize,
    interp: InterpFunc,
    input:  Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex<f32>>,
}

struct FFTResult {
    values: HashMap<u32, f32>,
    lookup: Vec<u32>,
    interp: InterpFunc,
}

struct SampleBuilder {
    is_mono: bool,
    channel: usize,
}

impl AudioIndex {
    fn new(start_time: f32, stop_time: f32, steps: usize, sampling_frequency: usize) -> Self {
        let start_frame: usize = (sampling_frequency as f32 * start_time) as usize;
        let stop_frame:  usize = (sampling_frequency as f32 * stop_time) as usize;
        let step_size:   usize = (stop_frame - start_frame) / steps as usize;
        AudioIndex {
            start_frame: start_frame,
            stop_frame: stop_frame,
            step_size: step_size,
            current_step: 0
        }
    }

    fn get_next_frame(&mut self) -> usize {
        let next = self.current_step * self.step_size + self.start_frame;
        self.current_step += 1;
        next
    }
}

impl FFTBuilder {
    fn new(
        width: usize,
        decimations: u8,
        window_func: WindowFunctionType,
        sample_builder: SampleBuilder,
        interp: InterpFunc,
        sampling_frequency: usize
        ) -> Self {
        let mut planner = FFTplanner::new(false);
        let buffer_size = width * usize::pow(2, decimations as u32);
        FFTBuilder {
            fft: planner.plan_fft(width),
            builder: sample_builder,
            decimations: decimations,
            sampling_frequency,
            fft_width: width,
            interp,
            buffer_size,
            input: vec![Zero::zero(); width],
            output: vec![Zero::zero(); width],
            window: window_func(width),
            buffer: vec![Zero::zero(); buffer_size],
        }
    }


    fn load_buffer(&mut self, wav_iter: &mut WaveFileIterator) {
        self.buffer.clear();
        for i in 0..self.buffer_size {
            self.buffer.push(self.builder.get_sample(wav_iter));
        }
    }

    fn process(&mut self, wav_iter: &mut WaveFileIterator, reference_frame: usize) -> FFTResult {
        let skip: usize = max(reference_frame as i32 - (self.buffer_size as i32 / 2), 0) as usize;
        wav_iter.skip(skip);
        let mut result = FFTResult::new(self.interp);
        let nyquist = self.sampling_frequency / 2;
        self.load_buffer(wav_iter);
        for i in 0..=self.decimations {
            // process fft
            let offset = (self.buffer.len() - self.fft_width) / 2;
            self.input.copy_from_slice(&self.buffer[offset..offset+self.fft_width]);
            self.fft.process(&mut self.input, &mut self.output);
            // add results
            let half_width = self.fft_width / 2;
            for i in 0..=(half_width) {
                let freq = i * nyquist / half_width;
                result.add_result(freq as u32, self.output.get(i).unwrap().re);
            }
            // TODO
            // half pass
            // decimate
        }
        result
    }
}

impl FFTResult {
    fn new(interp: InterpFunc) -> Self {
        FFTResult {
            values: HashMap::new(),
            lookup: Vec::new(),
            interp,
        }
    }

    fn add_result(&mut self, frequency: u32, amplitude: f32) {
        if self.values.get(&frequency) == None {
            self.values.insert(frequency, amplitude);
            self.lookup.push(frequency);
        }
    }

    fn finalize(&mut self) {
        self.lookup.sort_unstable();
    }

    fn get_frequency(&self, frequency: u32) -> f32 {
        let f = self.values.get(&frequency);
        match f {
            Some(result) => *result,
            None => self.interpolate(frequency),
        }
    }

    fn interpolate(&self, frequency: u32) -> f32 {
        match self.interp {
            InterpFunc::Linear => self.linear(frequency),
            InterpFunc::Quadratic => self.quadtratic(frequency),
            InterpFunc::Cubic => self.cubic(frequency),
        }
    }

    fn linear(&self, frequency: u32) -> f32 {
        let index = match self.lookup.binary_search(&frequency) {
            Ok(_) => panic!("Found value when not expected"),
            Err(index) => index,
        };
        let lower = self.lookup.iter().nth(index - 1).unwrap();
        let lower_value = self.values.get(lower).unwrap();
        let upper = self.lookup.iter().nth(index).unwrap();
        let upper_value = self.values.get(upper).unwrap();
        let weight: f32 = (frequency as f32 - *lower as f32) / ((upper - lower) as f32);
        interpolation::lerp(lower_value, upper_value, &weight)
    }

    fn quadtratic(&self, frequency: u32) -> f32 {
        // TODO
        // interpolation::quad_bez
        0.0
    }

    fn cubic(&self, frequency: u32) -> f32 {
        // TODO
        // interpolation::cub_bez
        0.0
    }
}

impl SampleBuilder {
    fn new(channel: usize, is_mono: bool) -> Self {
        SampleBuilder {
            channel,
            is_mono,
        }
    }

    fn get_sample(&self, wav_iter: &mut WaveFileIterator) -> Complex<f32> {
        let sample = wav_iter.next().unwrap();
        if self.is_mono {
            Complex::new(self.avg(&sample) as f32, 0.0)
        } else {
            Complex::new(sample[self.channel] as f32, 0.0)
        }
    }

    fn avg(&self, v: &Vec<i32>) -> i32 {
        let x: i32 = v.iter().sum();
        let y: i32 = v.len() as i32;
        x / y
    }
}

fn initialize_index(audio_section: &HashMap<String, String>, wav: &WaveFile, image_width: usize) -> AudioIndex {
    let start_time: f32 = audio_section.get(MUSIC_START_TIME).unwrap().parse().unwrap();
    let stop_time: f32 = audio_section.get(MUSIC_STOP_TIME).unwrap().parse().unwrap();
    AudioIndex::new(start_time, stop_time, image_width, wav.sample_rate())
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

fn window_rectangle(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_triangular(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_parzen(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_welch(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_sine(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_hann(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_hamming(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_blackman(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_nuttall(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_blackman_nuttall(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_blackman_harris(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement
fn window_flat_top(n: usize) -> Vec<f32> {
    vec![One::one(); n]
}

// TODO implement matching to functions
fn get_window_function(fft_section: &HashMap<String, String>) -> WindowFunctionType {
    let window_function = fft_section.get(FFT_WINDOW_FUNCTION).unwrap();
    match &window_function[..] {
        "rectangular"      => window_rectangle,
        "triangular"       => window_triangular,
        "parzen"           => window_parzen,
        "welch"            => window_welch,
        "sine"             => window_sine,
        "hann"             => window_hann,
        "hamming"          => window_hamming,
        "blackman"         => window_blackman,
        "nuttall"          => window_nuttall,
        "blackman-nuttall" => window_blackman_nuttall,
        "blackman-harris"  => window_blackman_harris,
        "flat-top"         => window_flat_top,
        _                  => panic!("Unknown Window function"),
    }
}

fn initialize_fft_builder(fft_section: &HashMap<String, String>, audio_section: &HashMap<String, String>, wav: &WaveFile) -> FFTBuilder {
    let fft_decimations: u8 = fft_section.get(FFT_DECIMATIONS).unwrap().parse().unwrap();
    let fft_width: usize    = fft_section.get(FFT_WIDTH).unwrap().parse().unwrap();
    let window_function     = get_window_function(fft_section);
    let channel: usize      = audio_section.get(MUSIC_CHANNEL).unwrap().parse().unwrap();
    let is_mono: bool       = audio_section.get(MUSIC_IS_MONO).unwrap().parse().unwrap();
    let sample_builder      = SampleBuilder::new(channel, is_mono);
    let interp_func         = &fft_section.get(FFT_INTERPOLATION).unwrap()[..];
    let interp_func = match interp_func {
        "linear" => InterpFunc::Linear,
        "quad"   => InterpFunc::Quadratic,
        "cubic"  => InterpFunc::Cubic,
        _        => panic!("Unknown Interpolation function"),
    };
    FFTBuilder::new(fft_width, fft_decimations, window_function, sample_builder, interp_func, wav.sample_rate())
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let conf = Ini::load_from_file(&args[1]).unwrap();

    let audio_section = conf.section(Some(SECTION_MUSIC)).unwrap();
    let image_section = conf.section(Some(SECTION_IMAGE)).unwrap();
    let fft_section = conf.section(Some(SECTION_FFT)).unwrap();

    let wav = load_audio_file(audio_section);
    let img = create_image(image_section);
    let gradient = load_gradient_file(image_section);
    let mut index = initialize_index(audio_section, &wav, img.width() as usize);
    let mut builder = initialize_fft_builder(fft_section, audio_section, &wav);
    let mut buffer: Vec<Vec<f32>> = vec![Vec::new(); img.width() as usize];

    for _i in 0..img.width() as i32 {
        let mut iter = wav.iter();
        let next_index = index.get_next_frame();
        let result = builder.process(&mut iter, next_index);
        println!("{:?}", result.get_frequency(200));
        for y in 0..img.height() as i32 {
            // TODO get f for each y pixel
            // insert into vec x
        }
    }

    // TODO identify upper limit
    // scale all values accordingly
    // plot image

    let output_file = image_section.get(IMAGE_OUTPUT_FILE).unwrap();
    img.save(output_file).unwrap();
}
