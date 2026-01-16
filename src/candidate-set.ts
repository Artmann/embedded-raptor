import invariant from 'tiny-invariant'

/**
 * A fixed-size collection that maintains the top N highest-value entries.
 * Uses a min-heap internally for O(log n) insertion when at capacity.
 */
export class CandidateSet {
  public readonly size: number
  private readonly heap: CandidateSetEntry[] = []

  constructor(size = 5) {
    invariant(size > 0, 'Size must be a positive integer.')

    this.size = size
  }

  add(key: string, value: number): void {
    invariant(key, 'Key must be provided.')
    invariant(value, 'Value must be provided.')

    if (this.heap.length < this.size) {
      // Under capacity: push and bubble up
      this.heap.push(new CandidateSetEntry(key, value))
      this.bubbleUp(this.heap.length - 1)
      return
    }

    // At capacity: compare with minimum (root of min-heap)
    if (value > this.heap[0].value) {
      // Replace root and bubble down
      this.heap[0] = new CandidateSetEntry(key, value)
      this.bubbleDown(0)
    }
  }

  count(): number {
    return this.heap.length
  }

  getEntries(): CandidateSetEntry[] {
    // Return sorted copy (highest to lowest)
    return this.heap.slice().sort((a, b) => b.value - a.value)
  }

  getKeys(): string[] {
    return this.getEntries().map((entry) => entry.key)
  }

  /**
   * Bubble up element at index to maintain min-heap property
   */
  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2)
      if (this.heap[parentIndex].value <= this.heap[index].value) {
        break
      }
      this.swap(parentIndex, index)
      index = parentIndex
    }
  }

  /**
   * Bubble down element at index to maintain min-heap property
   */
  private bubbleDown(index: number): void {
    const length = this.heap.length

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const leftChild = 2 * index + 1
      const rightChild = 2 * index + 2
      let smallest = index

      if (
        leftChild < length &&
        this.heap[leftChild].value < this.heap[smallest].value
      ) {
        smallest = leftChild
      }

      if (
        rightChild < length &&
        this.heap[rightChild].value < this.heap[smallest].value
      ) {
        smallest = rightChild
      }

      if (smallest === index) {
        break
      }

      this.swap(index, smallest)
      index = smallest
    }
  }

  private swap(i: number, j: number): void {
    const temp = this.heap[i]
    this.heap[i] = this.heap[j]
    this.heap[j] = temp
  }
}

class CandidateSetEntry {
  public readonly key: string
  public readonly value: number

  constructor(key: string, value: number) {
    this.key = key
    this.value = value
  }
}
