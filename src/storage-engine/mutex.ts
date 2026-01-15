/**
 * Simple async mutex for serializing write operations.
 *
 * Uses a FIFO queue to ensure fairness - waiters are processed
 * in the order they called acquire().
 */
export class Mutex {
  private locked = false
  private waiting: Array<() => void> = []

  /**
   * Acquire the mutex. If already locked, waits until released.
   */
  async acquire(): Promise<void> {
    if (!this.locked) {
      this.locked = true
      return
    }

    // Wait in queue until released
    return new Promise<void>((resolve) => {
      this.waiting.push(resolve)
    })
  }

  /**
   * Release the mutex. If there are waiters, the next one acquires.
   */
  release(): void {
    if (this.waiting.length > 0) {
      // Give lock to next waiter (FIFO)
      const next = this.waiting.shift()
      if (next) {
        next()
      }
    } else {
      this.locked = false
    }
  }

  /**
   * Check if the mutex is currently locked.
   */
  isLocked(): boolean {
    return this.locked
  }
}
