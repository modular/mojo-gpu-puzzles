## Key concepts

In this puzzle, you'll learn about:

- Basic GPU kernel structure
- Thread indexing with `thread_idx.x`
- Simple parallel operations

- **Parallelism**: Every thread runs the same kernel body concurrently
- **Thread indexing**: Access element at position `i = thread_idx.x`
- **Memory access**: Read from `a[i]` and write to `output[i]`
- **Data independence**: Each output depends only on its corresponding input

## Code to complete

```mojo
{{#include ../../../problems/p01/p01.mojo:add_10}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.mojo" class="filename">View full file: problems/p01/p01.mojo</a>

> **Note**: The skeleton declares `var i = thread_idx.x` for you, but nothing
> uses `i` until you fill in the kernel. Until then the compiler warns
> `assignment to 'i' was never used; assign to '_' instead?`. That's expected,
> and it clears as soon as your line reads from `a[i]` and writes to
> `output[i]`. Later puzzles scaffold their variables the same way.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `thread_idx.x` in `i`
2. Add 10 to `a[i]`
3. Store result in `output[i]`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p01
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p01
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p01
```

  </div>
  <div class="tab-content">

```bash
uv run poe p01
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p01/p01.mojo:add_10_solution}}
```

<div class="solution-explanation">

This solution:

- Gets thread index with `i = thread_idx.x`
- Adds 10 to input value: `output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0`

</div>
</details>
