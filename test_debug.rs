use std::sync::atomic::{AtomicU8, Ordering};
use bitflags::bitflags;

bitflags\! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProcessState: u8 {
        const RUNNING = 1 << 0;
        const STOPPED = 1 << 1;
        const ZOMBIE  = 1 << 2;
    }
}

fn main() {
    let state = AtomicU8::new(ProcessState::RUNNING.bits());
    
    // Simulate what happens during concurrent transitions
    println\!("Initial: {:08b}", state.load(Ordering::Acquire));
    
    // Thread 1: transition_to_stopped
    let result = state.fetch_update(Ordering::Release, Ordering::Relaxed, |curr| {
        let mut flags = ProcessState::from_bits_truncate(curr);
        println\!("Thread 1 sees: {:?} ({:08b})", flags, curr);
        if flags.contains(ProcessState::ZOMBIE) || \!flags.contains(ProcessState::RUNNING) {
            None
        } else {
            flags.remove(ProcessState::RUNNING);
            flags.insert(ProcessState::STOPPED);
            println\!("Thread 1 setting to: {:?} ({:08b})", flags, flags.bits());
            Some(flags.bits())
        }
    });
    
    println\!("After T1: {:08b}, result: {:?}", state.load(Ordering::Acquire), result);
    
    // Check the state
    let bits = state.load(Ordering::Acquire);
    let flags = ProcessState::from_bits_truncate(bits);
    println\!("Checking: running={}, stopped={}, zombie={}",
        flags.contains(ProcessState::RUNNING),
        flags.contains(ProcessState::STOPPED),
        flags.contains(ProcessState::ZOMBIE));
}
