//! Memory probe: build N vertices (and optionally E edges) and report resident
//! footprint, to find where this in-memory model tops out and extrapolate toward
//! a billion nodes. Run under the OS peak-RSS timer:
//!   cargo build --release --example mem_probe
//!   /usr/bin/time -l ./target/release/examples/mem_probe 8000000 4
//! Args: <vertices> [edges_per_vertex]. Steady-state RSS is sampled in-process
//! (mach task_info) after the builder is dropped; peak RSS (incl. the build
//! transient) comes from /usr/bin/time -l.

use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

// Resident set size of this process, in bytes, via mach task_basic_info.
#[cfg(target_os = "macos")]
fn rss_bytes() -> u64 {
    // MACH_TASK_BASIC_INFO = 20; count = 10 (u32 words of the struct).
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    let mut info = MachTaskBasicInfo {
        virtual_size: 0,
        resident_size: 0,
        resident_size_max: 0,
        user_time: [0; 2],
        system_time: [0; 2],
        policy: 0,
        suspend_count: 0,
    };
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
    let rc = unsafe {
        task_info(
            mach_task_self(),
            20,
            &mut info as *mut _ as *mut i32,
            &mut count,
        )
    };
    if rc == 0 {
        info.resident_size
    } else {
        0
    }
}

#[cfg(not(target_os = "macos"))]
fn rss_bytes() -> u64 {
    0
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % n as u64) as usize
    }
}

fn build(n: usize, eper: usize) -> Graph {
    let mut b = Builder::default();
    for i in 0..n {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![("age".to_string(), Value::Num((18 + (i % 62)) as f64))],
        ));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for i in 0..n {
        for _ in 0..eper {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("p{}", rng.below(n)),
                "KNOWS".to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let eper: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let base = rss_bytes();
    let g = build(n, eper);
    let after = rss_bytes();

    let v = g.vertex_count() as u64;
    let e = g.edge_count() as u64;
    let used = after.saturating_sub(base);
    println!("vertices {v}  edges {e}");
    println!("steady RSS {:.2} GB  (delta {:.2} GB)", gb(after), gb(used));
    if v > 0 {
        println!("  {:.0} bytes/vertex", used as f64 / v as f64);
    }
    if e > 0 {
        let v_part = v as f64 * 160.0; // rough vertex share, refined by the 0-edge run
        println!(
            "  ~{:.0} bytes/edge (after subtracting ~160 B/vertex)",
            (used as f64 - v_part) / e as f64
        );
    }
    // Extrapolate the vertex cost to 1e9.
    if eper == 0 && v > 0 {
        let per = used as f64 / v as f64;
        println!(
            "  → 1e9 vertices ≈ {:.0} GB at this rate",
            per * 1e9 / 1024.0 / 1024.0 / 1024.0
        );
    }
    std::hint::black_box(&g);
}
