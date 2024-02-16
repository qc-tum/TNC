#[derive(Debug, PartialEq, Eq)]
pub struct Connectivity {
    pub connectivity: Vec<(usize, usize)>,
    name: ConnectivityLayout,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConnectivityLayout {
    Condor,
    Eagle,
    Osprey,
    Sycamore,
}

impl Connectivity {
    /// Struct that defines connectivity of IBM device
    ///
    /// # Arguments
    ///
    /// * `str` - &str name of IBM device
    ///
    /// # Examples
    /// ```
    /// #use tensorcontraction::circuits::connectivity::Connectivity;
    /// #use tensorcontraction::circuits::connectivity::ConnectivityLayout;
    /// let cn = Connectivity::new(ConnectivityLayout::Eagle);
    /// ```
    pub fn new(name: ConnectivityLayout) -> Self {
        let connectivity = match name {
            ConnectivityLayout::Condor => condor_connect(),
            ConnectivityLayout::Eagle => eagle_connect(),
            ConnectivityLayout::Osprey => osprey_connect(),
            ConnectivityLayout::Sycamore => sycamore_connect(),
        };
        Self { connectivity, name }
    }
}

fn sycamore_connect() -> Vec<(usize, usize)> {
    vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (0, 14),
        (4, 15),
        (8, 16),
        (12, 17),
        (14, 18),
        (15, 22),
        (16, 26),
        (17, 30),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (30, 31),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (19, 29),
        (20, 31),
        (21, 31),
        (22, 32),
        (23, 33),
        (24, 34),
        (25, 35),
        (26, 36),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),
        (32, 33),
        (33, 34),
        (34, 35),
        (35, 36),
        (29, 37),
        (30, 38),
        (31, 39),
        (32, 40),
        (33, 41),
        (34, 42),
        (35, 43),
        (37, 38),
        (38, 39),
        (39, 40),
        (40, 41),
        (41, 42),
        (42, 43),
        (38, 44),
        (39, 45),
        (40, 46),
        (41, 47),
        (42, 48),
        (44, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        (45, 49),
        (46, 50),
        (47, 51),
        (49, 50),
        (50, 51),
        (50, 52),
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
