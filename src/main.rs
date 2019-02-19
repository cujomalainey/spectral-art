use wavefile::WaveFile;

fn main() {
    println!("Hello, world!");
    let f = WaveFile::open("./fixtures/test-s24le.wav").unwrap();
    println!("{:}", f.sample_rate());
}
