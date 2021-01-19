# A story about conjugate gradient in rust

Conjugate gradients (CG) is an iterative method for solving \\(Ax =
b\\). It is limited to the case where the matrix \\(A\\) is positive
semi definite (essentially, only scales vectors differently in
different directions). By iterative method we mean a simple way to
improve a solution, which we apply repeatedly starting from some
initial value.

This is the kind of numerical algorithm that Julia excels at, and XXX
some guy link XXX used it to demonstrate how Julia iterables are
wonderful way to express iterative algorithms, and also "utilities"
useful across algorithms, like timing, skipping, stopping conditions
etc.

I like Rust quite a bit, it abstracts loops via iterators and custom
abstractions are efficient. Could such a vocabulary work for rust as
well? to avoid extranous distractions, I decided to follow in their
footsteps, and essentially transcribed their implementation.

My implementation ended up looking like this:

```rust
        let ap = self.a.dot(&self.p).into_shared();
        let alpha = self.rs / self.p.dot(&ap);
        self.x += &(alpha * &self.p);
        self.r -= &(alpha * &ap);
        self.rsprev = self.rs;
        self.rs = self.r.dot(&self.r);
        self.p = (&self.r + &(&self.rs / self.rsprev * &self.p)).into_shared();
        self.ap = Some(ap);
```

which is quite similar to their Julia version (itself very similar to
the wikipedia rendition), except for being more explicit about memory
usage. Oh, and except for the bug.

To start, all seemed well. Following IMDR, we define a struct
`CGIterable` whose fields hold almost all intermediate state of the
algorithm. A function takes a problem (in this case, \\(A,b\\)) and
returns a CGIterable, for which we implement the StreamingIterator
trait. This trait is simple, and requires us to implement two methods:
`advance` (apply the code above to run an iteration of the algorithm)
and `get` (borrowing `self`, return `Some(self)`). `get` returning the
struct by reference is the main motivation to use StreamingIterator
over the ubiquitous Iterator trait; it leaves any copying up to the
implementor.

The signatures for implementing a method is as follows:
```rust
/// The state of a conjugate gradient algorithm.
#[derive(Clone)]
struct CGIterable {
    // contents
}

impl CGIterable {
    /// Convert a LinearSystem problem into a StreamingIterator of conjugate gradient solutions.
    pub fn conjugate_gradient(problem: LinearSystem) -> CGIterable {
	    // we chose to consume the LinearSystem, but it can probably be borrowed instead
    }
}

impl StreamingIterator for CGIterable {
    type Item = CGIterable;
    /// Implementation of conjugate gradient iteration
    fn advance(&mut self) {
	    // the code from before
    }
    fn get(&self) -> Option<&Self::Item> {
	    // We will eventually change this, but for now:
        Some(self)
    }
}
```

To use it, we initialize the iterator and soon start reading values:

```rust
let p = make_3x3_psd_system_2();
let cg_iter = CGIterable::conjugate_gradient(p);
while let Some(_cgi) = cg_print_iter.next() {
	let res = result.a.dot(&result.x) - &result.b;
	let res_norm = res.dot(&res);
    println!(
        "||Ax - b ||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
        res_norm,
		result.x,
		result.a.dot(&result.x) - &result.b,
}
```

Indeed the output shows nice convergence, with the discrepancy \\(Ax -
b\\) tending quickly to zero:

```
||Ax - b ||_2 = 1.00000, for x = [0.0000, 1.0000, 0.0000], and Ax - b = [0.50000, 0.00000, 0.50000]
||Ax - b ||_2 = 0.06250, for x = [-0.6250, 1.3125, -0.6250], and Ax - b = [0.03125, 0.00000, 0.03125]
||Ax - b ||_2 = 0.00391, for x = [-0.6641, 1.3320, -0.6641], and Ax - b = [0.00195, 0.00000, 0.00195]
||Ax - b ||_2 = 0.00024, for x = [-0.6665, 1.3333, -0.6665], and Ax - b = [0.00012, 0.00000, 0.00012]
||Ax - b ||_2 = 0.00002, for x = [-0.6667, 1.3333, -0.6667], and Ax - b = [0.00001, 0.00000, 0.00001]
```

Ok, while we have nicely abstracted the algorithm itself to the call
to `conjugate_gradient`, our loop now contains quite a bit of code to
merely report progress. It is also an infinite loop; a stopping
condition would be additional code, and other common niceties include:

- looking at only every Nth iteration,
- time the cost of an iteration (but not auxiliary I/O!), 
- plot progress,
- save progress, etc. 

Can we abstract all these so that we can reuse them for other methods
and solve each problem once? and where is that bug? how does one even
catch bugs in code that doesn't ever really finish?

All those and more, next time.
