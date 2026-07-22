# M-ABD 10x10 ball-joint net

A focused proof-of-concept implementation of the ball-joint net from Figure 12
of *M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics*.

The scene contains the paper's inferred 280-body topology:

- 100 affine hub bodies arranged in a 10x10 grid
- 180 affine rod bodies joining horizontal and vertical neighbors
- 360 ball joints, or 1,080 scalar positional constraints
- 10 fixed hub bodies along the top boundary

Each affine body is represented by the four control points from Section 4.1.
Each fixed step performs an implicit prediction and a compact co-rotated
local/global solve, using a matrix-free dual KKT solve for all linear ball-joint
constraints. The demo uses the Figure 12 timestep of `1/30 s`. A compact overlay
reports smoothed FPS and frame time, the measured simulation-step duration,
simulation rate, and scene size.

## Run natively

```sh
cargo run
```

## Run in a browser

Install the one-time prerequisites if needed:

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
```

Start the development server:

```sh
trunk serve --open
```

The site is available at <http://127.0.0.1:8080> if it does not open
automatically. Create a deployable build with:

```sh
trunk build --release
```

## MVP scope

This demo includes only the paper mechanics needed for Figure 12: co-rotated
affine bodies, fixed-step implicit prediction, linear ball joints, and a dual
constraint solve. It intentionally omits contacts, self-collision, other joint
types, controls, GPU compute, and the paper's million-body solver optimizations.
