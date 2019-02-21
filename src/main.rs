extern crate num;

use wavefile::WaveFile;

fn avg(v: &Vec<i32>) -> i32 {
    let x: i32 = v.iter().sum();
    let y: i32 = v.len() as i32;
    x / y
}

fn main() {
    println!("Hello, world!");
    let f = WaveFile::open("holdon.wav").unwrap();
    println!("{:}", f.sample_rate());
    let mut iter = f.iter();
    let frame = iter.nth(0).unwrap();

    println!("Data is {:?}", avg(&frame));
}
