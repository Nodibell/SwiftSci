import Foundation

// First load occurs after workspace admission and metadata validation.
actor PersistentAdmissionTiming {
    let start: ContinuousClock.Instant
    private var wait: Double?
    var waitSeconds: Double { wait ?? 0 }
    init(start: ContinuousClock.Instant) { self.start = start }
    func admitted() {
        if wait == nil { wait = elapsedSeconds(since: start) }
    }
}
