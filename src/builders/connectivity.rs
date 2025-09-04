use itertools::Itertools;

/// Struct that defines connectivity of IBM device.
#[derive(Debug, PartialEq, Eq)]
pub struct Connectivity {
    pub connectivity: Vec<(usize, usize)>,
    name: ConnectivityLayout,
}

/// Different types of connectivity layouts of IBM devices.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConnectivityLayout {
    Condor,
    Eagle,
    Osprey,
    Sycamore,
    All(usize),
    Line(usize),
}

impl Connectivity {
    /// Create a new connectivity layout instance.
    ///
    /// # Arguments
    ///
    /// * `str` - &str name of IBM device
    ///
    /// # Examples
    /// ```
    /// # use tnc::builders::connectivity::Connectivity;
    /// # use tnc::builders::connectivity::ConnectivityLayout;
    /// let cn = Connectivity::new(ConnectivityLayout::Eagle);
    /// ```
    #[must_use]
    pub fn new(name: ConnectivityLayout) -> Self {
        let connectivity = match name {
            ConnectivityLayout::Condor => condor_connect(),
            ConnectivityLayout::Eagle => eagle_connect(),
            ConnectivityLayout::Osprey => osprey_connect(),
            ConnectivityLayout::Sycamore => sycamore_connect(),
            ConnectivityLayout::All(n) => all_connect(n),
            ConnectivityLayout::Line(n) => line_connect(n),
        };
        Self { connectivity, name }
    }
}

fn all_connect(n: usize) -> Vec<(usize, usize)> {
    assert!(n > 0);

    let mut v = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            v.push((i, j));
        }
    }
    v
}

fn sycamore_connect() -> Vec<(usize, usize)> {
    vec![
        (52, 32),
        (32, 31),
        (31, 24),
        (24, 29),
        (29, 26),
        (26, 40),
        (40, 44),
        (44, 53),
        (37, 32),
        (32, 21),
        (21, 24),
        (24, 18),
        (18, 26),
        (26, 25),
        (25, 44),
        (44, 48),
        (37, 22),
        (22, 21),
        (21, 7),
        (7, 18),
        (18, 15),
        (15, 25),
        (25, 42),
        (42, 48),
        (35, 22),
        (22, 8),
        (8, 7),
        (7, 5),
        (5, 15),
        (15, 16),
        (16, 42),
        (42, 46),
        (35, 11),
        (11, 8),
        (8, 1),
        (1, 5),
        (5, 6),
        (6, 16),
        (16, 51),
        (51, 46),
        (11, 4),
        (4, 1),
        (1, 2),
        (2, 6),
        (6, 12),
        (12, 51),
        (51, 47),
        (14, 4),
        (4, 3),
        (3, 2),
        (2, 10),
        (10, 12),
        (12, 41),
        (41, 47),
        (36, 14),
        (14, 13),
        (13, 3),
        (3, 9),
        (9, 10),
        (10, 20),
        (20, 41),
        (41, 50),
        (36, 27),
        (27, 13),
        (13, 17),
        (17, 9),
        (9, 19),
        (19, 20),
        (20, 43),
        (43, 50),
        (38, 27),
        (27, 28),
        (28, 17),
        (17, 23),
        (23, 19),
        (19, 34),
        (34, 43),
        (43, 49),
        (38, 39),
        (39, 28),
        (28, 30),
        (30, 23),
        (23, 33),
        (33, 34),
        (34, 45),
        (45, 49),
    ]
}

pub(super) fn sycamore_a() -> Vec<(usize, usize)> {
    vec![
        (31, 32),
        (29, 24),
        (40, 26),
        (53, 44),
        (21, 22),
        (18, 7),
        (25, 15),
        (48, 42),
        (8, 11),
        (5, 1),
        (16, 6),
        (46, 51),
        (14, 4),
        (2, 3),
        (12, 10),
        (47, 41),
        (13, 27),
        (9, 17),
        (20, 19),
        (50, 43),
        (28, 39),
        (23, 30),
        (34, 33),
        (49, 45),
    ]
}

pub(super) fn sycamore_b() -> Vec<(usize, usize)> {
    vec![
        (32, 37),
        (24, 21),
        (26, 18),
        (44, 25),
        (22, 35),
        (7, 8),
        (15, 5),
        (42, 16),
        (1, 4),
        (6, 2),
        (51, 12),
        (14, 36),
        (3, 13),
        (10, 9),
        (41, 20),
        (27, 38),
        (17, 28),
        (19, 23),
        (43, 34),
    ]
}

pub(super) fn sycamore_c() -> Vec<(usize, usize)> {
    vec![
        (52, 32),
        (31, 24),
        (29, 26),
        (40, 44),
        (37, 22),
        (21, 7),
        (18, 15),
        (25, 42),
        (35, 11),
        (8, 1),
        (5, 6),
        (16, 51),
        (4, 3),
        (2, 10),
        (12, 41),
        (36, 27),
        (13, 17),
        (9, 19),
        (20, 43),
        (38, 39),
        (28, 30),
        (23, 33),
        (34, 45),
    ]
}

pub(super) fn sycamore_d() -> Vec<(usize, usize)> {
    vec![
        (32, 21),
        (24, 18),
        (26, 25),
        (44, 48),
        (22, 8),
        (7, 5),
        (15, 16),
        (42, 46),
        (11, 4),
        (1, 2),
        (6, 12),
        (51, 47),
        (14, 13),
        (3, 9),
        (10, 20),
        (41, 50),
        (27, 28),
        (17, 23),
        (19, 34),
        (43, 49),
    ]
}

#[allow(
    dead_code,
    reason = "Might be needed for more advanced Sycamore test cases"
)]
pub(super) fn sycamore_e() -> Vec<(usize, usize)> {
    vec![
        (52, 32),
        (29, 26),
        (24, 18),
        (21, 7),
        (22, 8),
        (35, 11),
        (44, 48),
        (25, 42),
        (15, 16),
        (5, 6),
        (1, 2),
        (4, 3),
        (14, 13),
        (36, 27),
        (51, 47),
        (12, 41),
        (10, 20),
        (9, 19),
        (17, 23),
        (28, 30),
        (43, 49),
        (34, 45),
    ]
}

#[allow(
    dead_code,
    reason = "Might be needed for more advanced Sycamore test cases"
)]
pub(super) fn sycamore_f() -> Vec<(usize, usize)> {
    vec![
        (31, 24),
        (32, 21),
        (37, 22),
        (40, 44),
        (26, 25),
        (18, 15),
        (7, 5),
        (8, 1),
        (11, 4),
        (42, 46),
        (16, 51),
        (6, 12),
        (2, 10),
        (3, 9),
        (13, 17),
        (27, 28),
        (38, 39),
        (41, 50),
        (20, 43),
        (19, 34),
        (23, 33),
    ]
}

#[allow(
    dead_code,
    reason = "Might be needed for more advanced Sycamore test cases"
)]
pub(super) fn sycamore_g() -> Vec<(usize, usize)> {
    vec![
        (27, 38),
        (28, 39),
        (14, 4),
        (13, 3),
        (17, 9),
        (23, 19),
        (33, 34),
        (37, 32),
        (22, 21),
        (8, 7),
        (5, 1),
        (6, 2),
        (12, 10),
        (41, 20),
        (50, 43),
        (29, 24),
        (26, 18),
        (25, 15),
        (42, 16),
        (46, 51),
        (53, 44),
    ]
}

#[allow(
    dead_code,
    reason = "Might be needed for more advanced Sycamore test cases"
)]
pub(super) fn sycamore_h() -> Vec<(usize, usize)> {
    vec![
        (14, 36),
        (13, 27),
        (17, 28),
        (23, 30),
        (22, 35),
        (8, 11),
        (1, 4),
        (2, 3),
        (10, 9),
        (20, 19),
        (43, 34),
        (49, 45),
        (31, 32),
        (24, 21),
        (18, 7),
        (15, 5),
        (16, 6),
        (51, 12),
        (47, 41),
        (40, 26),
        (44, 25),
        (48, 42),
    ]
}

fn hexagon_connectivity<I>(
    row_length: usize,
    connectivity: &mut Vec<(usize, usize)>,
    stride: usize,
    mut bridge_cycle_prev: I,
    bridges: usize,
    rows: usize,
) where
    I: Iterator<Item = usize> + Clone,
{
    let mut count = 0;
    let mut bridge_cycle_next = bridge_cycle_prev.clone();
    // first row
    let mut prev_last;
    let mut next_last = row_length - 2;
    for i in 0..(row_length - 2) {
        connectivity.push((i, i + 1));

        if i % stride == 0 {
            connectivity.push((i, next_last + bridge_cycle_prev.next().unwrap()));
        }
    }
    count += row_length - 1 + bridges;
    for row in 0..rows {
        //first push intermediate
        prev_last = next_last;
        next_last = count + row_length - 1;

        for i in 0..(row_length - 1) {
            if (i + 2 * (row % 2)) % 4 == 0 {
                connectivity.push((prev_last + bridge_cycle_prev.next().unwrap(), count + i));
            }

            connectivity.push((count + i, count + i + 1));

            if (i + 2 * ((row + 1) % 2)) % 4 == 0 {
                connectivity.push((count + i, next_last + bridge_cycle_next.next().unwrap()));
            }
        }
        if row % 2 == 0 {
            connectivity.push((next_last, next_last + bridge_cycle_next.next().unwrap()));
        }
        count += row_length + bridges;
    }

    prev_last = next_last;
    next_last = count + row_length - 2;

    for i in 0..(row_length - 2) {
        if (i + 3) % stride == 0 {
            connectivity.push((prev_last + bridge_cycle_next.next().unwrap(), count + i));
        }
        connectivity.push((count + i, count + i + 1));
    }
    //last row
    connectivity.push((prev_last + bridge_cycle_next.next().unwrap(), next_last));
}

fn eagle_connect() -> Vec<(usize, usize)> {
    let rows = 5;
    let row_length = 15;
    let bridge_cycle = (1..=4).cycle();
    let bridges = 4;
    let stride = 4;
    let mut connectivity = Vec::new();

    hexagon_connectivity(
        row_length,
        &mut connectivity,
        stride,
        bridge_cycle,
        bridges,
        rows,
    );

    connectivity
}

fn osprey_connect() -> Vec<(usize, usize)> {
    let rows = 11;
    let row_length = 27;
    let bridge_cycle = (1..=7).cycle();
    let bridges = 7;
    let stride = 4;
    let mut connectivity = Vec::new();

    hexagon_connectivity(
        row_length,
        &mut connectivity,
        stride,
        bridge_cycle,
        bridges,
        rows,
    );

    connectivity
}

fn condor_connect() -> Vec<(usize, usize)> {
    let rows = 19;
    let row_length = 43;
    let bridge_cycle = (1..=11).cycle();
    let bridges = 11;
    let stride = 4;
    let mut connectivity = Vec::new();

    hexagon_connectivity(
        row_length,
        &mut connectivity,
        stride,
        bridge_cycle,
        bridges,
        rows,
    );

    connectivity
}

fn line_connect(n: usize) -> Vec<(usize, usize)> {
    (0..n).tuple_windows().collect::<Vec<_>>()
}
