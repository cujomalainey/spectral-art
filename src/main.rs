use wavefile::WaveFile;

fn main() {
    println!("Hello, world!");
    let f = WaveFile::open("holdon.wav").unwrap();
    println!("{:}", f.sample_rate());
    let mut iter = f.iter();
    let frame = iter.nth(0).unwrap();
}
