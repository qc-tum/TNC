use std::{collections::HashMap, ops::RangeBounds, path::PathBuf, rc::Rc};

use tetra::{contract, Tensor as DataTensor};

use crate::{gates::load_gate, io::load_data, tensornetwork::Tensor, types::*};

use super::tensordata::TensorData;

/// Fully contracts a list of [DataTensor] objects based on a given contraction path using repeated SSA format.
///
/// # Arguments
///
/// * `tn` - [`Tensor`] to be contracted
/// * `d_tn` - [`Vector`] of [DataTensor] objects containing data of [Tensor]
/// * `contract_path` - [`Vector`] of [(usize, usize)], indicating contraction path. See [BranchBound] for details on `contract_path` format.
///
/// # Examples
///
/// ```
/// # extern crate tensorcontraction;
/// # use tensorcontraction::{
///     contractionpath::paths::{branchbound::BranchBound, CostType, OptimizePath},
///     random::tensorgeneration::random_tensor_network_with_rng,
///     tensornetwork::tensor::Tensor,
///     tensornetwork::contraction::contract_tensor_network,
/// };
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// let mut r = StdRng::seed_from_u64(42);
/// let mut r_tn = random_tensor_network_with_rng(2, 3, &mut r);
/// let mut opt = BranchBound::new(&r_tn, None, 20, CostType::Flops);
/// opt.optimize_path();
/// let opt_path = opt.get_best_replace_path();
/// contract_tensor_network(&mut r_tn, &opt_path);
/// ```
pub fn contract_tensor_network(tn: &mut Tensor, contract_path: &[ContractionIndex]) {
    let mut last_index = 0;
    for contract_index in contract_path.iter() {
        match contract_index {
            ContractionIndex::Pair(i, j) => {
                tn.contract_tensors(*i, *j);
                last_index = *i;
            }
            ContractionIndex::Path(i, inner_contract_path) => {
                contract_tensor_network(tn.get_mut_tensor(*i), inner_contract_path);
                tn._update_tensor(&mut tn.get_tensor(*i).clone());
            }
        }
    }
    tn.set_legs(tn.get_tensor(last_index).get_legs().clone());
    let tmp_data = tn.get_tensor(last_index).get_tensor_data().clone();
    tn.drain(0..);
    tn.set_tensor_data(tmp_data);
}

pub(crate) trait TensorContraction {
    /// Internal method to permute tensor
    fn get_mut_tensor(&mut self, i: usize) -> &mut Tensor;
    fn get_mut_edges(&mut self) -> &mut HashMap<EdgeIndex, Vec<Vertex>>;
    fn get_data(&self) -> DataTensor;
    fn swap(&mut self, i: usize, j: usize);
    fn drain<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>;
    fn contract_tensors(&mut self, tensor_a_loc: usize, tensor_b_loc: usize);
}

impl TensorContraction for Tensor {
    fn get_mut_tensor(&mut self, i: usize) -> &mut Tensor {
        &mut self.tensors[i]
    }

    // Internal method to update edges
    fn get_mut_edges(&mut self) -> &mut HashMap<EdgeIndex, Vec<Vertex>> {
        &mut self.edges
    }

    /// Getter for underlying raw data
    fn get_data(&self) -> DataTensor {
        match &*self.get_tensor_data() {
            TensorData::File(filename) => load_data(filename).unwrap(),
            TensorData::Gate((gatename, angles)) => load_gate(gatename, Some(angles)), // load_gate[gatename.to_lowercase()],
            TensorData::Matrix(rawdata) => rawdata.clone(),
            TensorData::Uncontracted => DataTensor::new(&[]),
        }
    }

    // Internal method to swap tensors
    fn swap(&mut self, i: usize, j: usize) {
        self.tensors.swap(i, j);
    }

    /// Drains the `tensor` vector. Mainly used to clear data after contraction.
    fn drain<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        self.tensors.drain(range);
    }

    fn contract_tensors(&mut self, tensor_a_loc: usize, tensor_b_loc: usize) {
        let tensor_a = self.get_mut_tensor(tensor_a_loc).clone();
        let tensor_b = self.get_mut_tensor(tensor_b_loc).clone();

        let tensor_a_legs = tensor_a.get_legs();
        let tensor_b_legs = tensor_b.get_legs();
        // let tensor_union = &tensor_b | &tensor_a;
        let tensor_symmetric_difference = &tensor_b ^ &tensor_a;

        // let counter = count_edges(tensor_union.get_legs().iter());

        let edges = self.get_mut_edges();
        // for leg in tensor_union.get_legs().unique().iter() {
        //     // Check if hyperedges are being contracted, if so, only append once to output tensor
        //     let mut i = 0;
        //     while edges[leg].len() - 1 > (counter[leg] + i) {
        //         i += 1;
        //         tensor_symmetric_difference.legs.push(*leg);
        //     }
        // }
        // Update internal edges HashMap to point tensor b legs to new contracted tensor
        for leg in tensor_b_legs.iter() {
            edges.entry(*leg).and_modify(|e| {
                e.retain(|e| {
                    if let Vertex::Closed(edge) = e {
                        *edge != tensor_a_loc
                    } else {
                        true
                    }
                });
                for edge in &mut e.iter_mut() {
                    if let Vertex::Closed(tensor_loc) = edge {
                        if *tensor_loc == tensor_b_loc {
                            *edge = Vertex::Closed(tensor_a_loc);
                        }
                    }
                }
            });
        }
        let mut new_tensor = Tensor::new(tensor_symmetric_difference.get_legs().clone());
        new_tensor.bond_dims = Rc::clone(&self.bond_dims);
        new_tensor.set_tensor_data(TensorData::Matrix(contract(
            tensor_symmetric_difference
                .get_legs()
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            tensor_a_legs
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            &self.get_tensor(tensor_a_loc).get_data(),
            tensor_b_legs
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            &self.get_tensor(tensor_b_loc).get_data(),
        )));
        self.tensors[tensor_a_loc] = new_tensor;
        // remove old tensor
        self.tensors[tensor_b_loc] = Tensor::new(Vec::new());
        // (tensor_intersect, tensor_difference)
    }
}

fn count_edges<I>(it: I) -> HashMap<I::Item, usize>
where
    I: IntoIterator,
    I::Item: Eq + core::hash::Hash,
{
    let mut result = HashMap::new();

    for item in it {
        *result.entry(item).or_insert(0) += 1;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::contract_tensor_network;
    use crate::{
        path,
        tensornetwork::{
            contraction::TensorContraction, create_tensor_network, tensor::Tensor,
            tensordata::TensorData,
        },
        types::Vertex,
    };

    use num_complex::Complex64;
    use std::collections::HashMap;
    use tetra::Layout;

    fn setup() -> (
        Vec<Complex64>,
        Vec<Complex64>,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let d1 = [
            0.69469607, 0.03142814, 0.56333184, 0.12908922, 0.68881492, 0.38906653, 0.28704775,
            0.66259172, 0.03017098, 0.0216769, 0.13239795, 0.56624022, 0.60589695, 0.52471058,
            0.08573655, 0.06819372, 0.1568983, 0.41559434, 0.6708583, 0.60417368, 0.98723314,
            0.25018858, 0.14947663, 0.70206464, 0.49755784, 0.50813521, 0.54859423, 0.62646753,
            0.95172281, 0.12807469, 0.56603429, 0.05893249, 0.64737241, 0.04312631, 0.80191274,
            0.17439514, 0.00265264, 0.4052311, 0.91152868, 0.88411605, 0.8787456, 0.88505868,
        ]
        .iter()
        .map(|e| Complex64::new(*e, 0.0))
        .collect();

        let d2 = [
            0.09328754, 0.56783732, 0.16813387, 0.82429821, 0.0184785, 0.73521183, 0.56154307,
            0.36866055, 0.93516298, 0.66265138, 0.75256279, 0.46568749, 0.14677414, 0.24352534,
            0.42248108, 0.4729148, 0.98754226, 0.65741335, 0.71002821, 0.36600333, 0.68610491,
            0.05782579, 0.18485907, 0.94518584, 0.95895765, 0.85841239, 0.05705296, 0.09616524,
            0.63222811, 0.87366903, 0.90875357, 0.47705938, 0.65134858, 0.6608632, 0.95581392,
            0.32332593, 0.53788954, 0.903023, 0.40153079, 0.37487737, 0.30883193, 0.53174817,
            0.39644565, 0.25535147, 0.58535347, 0.77200294, 0.86162035, 0.75814678, 0.72907822,
            0.19955574, 0.18811225, 0.26336036, 0.15323682, 0.26735153, 0.55806485, 0.05887273,
            0.18610468, 0.50636222, 0.45368943, 0.42390405, 0.96522828, 0.84946421, 0.40025282,
            0.19881981, 0.02848654, 0.33727215, 0.95483207, 0.77974469, 0.31792446, 0.39641724,
            0.18105829, 0.17672455, 0.93845113, 0.06969526, 0.02687802, 0.89507515, 0.48635865,
            0.32632963, 0.30049458, 0.097193, 0.98697788, 0.61149565, 0.00866831, 0.71612395,
            0.82291458, 0.49291276, 0.61479999, 0.94664182, 0.4173764, 0.99885288, 0.797537,
            0.11157382, 0.71116417, 0.49948545, 0.99517939, 0.14450441, 0.51148864, 0.35196431,
            0.62919375, 0.35710482, 0.51206311, 0.20117422, 0.88595471, 0.55022357, 0.74170559,
            0.29588465, 0.93177878, 0.25674534, 0.77183918, 0.46184645, 0.48719714, 0.62120151,
            0.50764396, 0.78361974, 0.65953757, 0.20511562, 0.73759098, 0.14039605, 0.83019634,
            0.8332694, 0.55324321, 0.10042969, 0.18699679, 0.18950935, 0.89248192, 0.74924386,
            0.92748739, 0.80220305, 0.47501941, 0.26003983, 0.95522559, 0.91094124, 0.6795759,
            0.41454924, 0.80801303, 0.59779671, 0.51192525, 0.91462506, 0.92084146, 0.61720142,
            0.57376556, 0.24997149, 0.57329336, 0.43957024, 0.50238366, 0.67373356, 0.11259408,
            0.37683318, 0.49300396, 0.6711206, 0.94882965, 0.59395913, 0.05114786, 0.73600206,
            0.00601697, 0.5727972, 0.67090141, 0.58077019, 0.64640978, 0.55276303, 0.06095278,
            0.07539359, 0.17735247, 0.82562252, 0.78428553, 0.1906632, 0.33509431, 0.42012132,
            0.00159857, 0.60167655, 0.24232731, 0.83693469, 0.03338013, 0.85122908, 0.40059668,
            0.71621368, 0.34265388, 0.09638051, 0.64427604, 0.23733484, 0.56653551, 0.52411397,
            0.00628771, 0.57258058, 0.16022726, 0.36959163, 0.75623427, 0.58954677, 0.82969614,
            0.87791391, 0.81419296, 0.9083103, 0.02033065, 0.33228088, 0.6885901, 0.70155266,
            0.41101974, 0.15976186, 0.23377142, 0.06092987, 0.45264005, 0.2132634, 0.09463183,
            0.29559498, 0.90050621, 0.90966579, 0.33135194, 0.75781385, 0.89580607, 0.89514179,
            0.56557729, 0.08695989, 0.64372738, 0.15244104, 0.61556173, 0.43501657, 0.83772869,
            0.95057826, 0.74261719, 0.79065485, 0.11257079, 0.40928239, 0.83167007, 0.89305546,
            0.38080999, 0.64519961, 0.86476415, 0.81232636, 0.52187504, 0.9641269, 0.87137541,
            0.33997589, 0.5117147, 0.97633497, 0.17706302, 0.72655448, 0.36950375, 0.73625114,
            0.99500737, 0.9711593, 0.88231438, 0.00364066, 0.63363962, 0.77629796, 0.76189465,
            0.79594656, 0.06046632, 0.37121335, 0.11968514, 0.34248849, 0.04004532, 0.12645287,
            0.27779065, 0.10250479, 0.24468091, 0.13638938, 0.3328005, 0.84108116, 0.13515933,
            0.65145451, 0.4327036, 0.05269428, 0.9111833, 0.69159617, 0.14994096, 0.79117789,
            0.78700038, 0.16870521, 0.78504774, 0.44292063, 0.62066272, 0.81990975, 0.9322563,
            0.28132287, 0.21205641, 0.7865617, 0.61846373, 0.8919656, 0.87968058, 0.27411728,
            0.38383478, 0.70716565, 0.2629087, 0.67097425, 0.69508973, 0.11427584, 0.92469918,
            0.69439856, 0.67802797, 0.85167929, 0.31096221, 0.92851952, 0.20776057, 0.91726347,
            0.90551286, 0.24317468, 0.48989766, 0.57365687, 0.56664079, 0.37988123, 0.84965442,
            0.04234216, 0.72334513, 0.90954789, 0.35181637, 0.64300879, 0.78047689, 0.88886375,
            0.66392189, 0.71558052, 0.33957753, 0.86017955, 0.24987067, 0.75239371, 0.846215,
            0.80888273, 0.25652923, 0.47976616, 0.71021803, 0.25107151, 0.25820616, 0.65540941,
            0.05381956, 0.31208421, 0.80435833, 0.51970853, 0.92648614, 0.93166913, 0.30201433,
            0.58696402, 0.73332323, 0.389746, 0.40119842, 0.93395815, 0.47877774, 0.61972036,
        ]
        .iter()
        .map(|e| Complex64::new(*e, 0.0))
        .collect();

        let d3 = [
            0.42667703, 0.48457094, 0.68925937, 0.99198568, 0.55416889, 0.38235533, 0.23278915,
            0.35852096, 0.58649561, 0.93958456, 0.64993388, 0.89260841, 0.50899537, 0.45536893,
            0.25893103, 0.61086187, 0.98899915, 0.91573474, 0.19398411, 0.68390863, 0.39154508,
            0.73696751, 0.5698224, 0.78365831, 0.46399883, 0.96999598, 0.96014711, 0.72481088,
            0.30197295, 0.32362527, 0.85744741, 0.03117345, 0.99394162, 0.73509025, 0.02588929,
            0.79422479, 0.19915962, 0.57046363, 0.74970349, 0.92350756, 0.90096793, 0.63684391,
            0.07446369, 0.25514523, 0.97320958, 0.87981084, 0.4379634, 0.63565715, 0.83553603,
            0.3833003, 0.0772783, 0.16907803, 0.15863019, 0.04340611, 0.15821493, 0.11871337,
            0.9943052, 0.24349509, 0.13325565, 0.15391281, 0.01091549, 0.28614178, 0.41929479,
            0.74795668, 0.40158558, 0.96101644, 0.49369888, 0.63394661, 0.80317387, 0.52994169,
            0.66743804, 0.76277137, 0.25734803, 0.90787102, 0.86028783, 0.9712097, 0.79091853,
            0.97385292, 0.28053365, 0.21851811, 0.21683468, 0.89221896, 0.14093243, 0.5011467,
            0.04057924, 0.2620503, 0.97840279, 0.35671425, 0.07766314, 0.38162112, 0.88516118,
            0.74285925, 0.86502545, 0.86329524, 0.59698018, 0.41562044, 0.16286828, 0.24323371,
            0.00842274, 0.19534713, 0.83349213, 0.89453333, 0.8861164, 0.8331781, 0.72101998,
            0.00771226, 0.21253237, 0.882532, 0.07932374, 0.59678746, 0.04757306, 0.71880735,
            0.91201475, 0.51059997, 0.51537244, 0.24251543, 0.80763707, 0.75864631, 0.37702641,
            0.79754793,
        ]
        .iter()
        .map(|e| Complex64::new(*e, 0.0))
        .collect();

        let dout = [
            21.12662191,
            22.54781684,
            19.14112552,
            21.50383642,
            20.41325586,
            21.21790454,
            19.02018308,
            19.80617065,
            18.14298459,
            19.23444656,
            22.09757373,
            23.84187648,
            20.99919352,
            21.00826462,
            17.48069436,
            20.77190844,
            18.61067063,
            20.64907039,
            17.89339839,
            19.46748844,
            18.55191736,
            18.47989775,
            21.51424258,
            23.15271771,
            21.58618595,
            21.87267004,
            19.59396099,
            22.66100042,
            19.69575675,
            21.82218509,
            19.89800678,
            21.31436564,
            19.46194771,
            19.86700794,
            22.58850997,
            24.51772651,
            22.31452844,
            23.62575299,
            20.01337021,
            22.48410562,
            21.13354029,
            22.92646174,
            20.28317164,
            20.82815715,
            19.11679716,
            21.63729977,
            23.81438329,
            25.89513636,
            25.95290253,
            27.8205038,
            23.56074292,
            27.02466869,
            24.12486376,
            26.50415702,
            24.02478861,
            25.81369869,
            22.49520341,
            24.56743911,
            26.9066631,
            30.23243803,
        ]
        .iter()
        .map(|e| Complex64::new(*e, 0.0))
        .collect();

        (d1, d2, d3, dout)
    }

    #[test]
    fn test_tn_contraction() {
        // t1 is of shape [3, 2, 7]
        let mut t1 = Tensor::new(vec![0, 1, 2]);

        // t2 is of shape [7, 8, 6]
        let mut t2 = Tensor::new(vec![2, 3, 4]);
        // t3 is of shape [3, 5, 8]
        let mut t3 = Tensor::new(vec![0, 5, 3]);
        // tout is of shape [5, 6, 2]
        let mut tout = Tensor::new(vec![5, 4, 1]);
        let bond_dims = HashMap::from([(0, 3), (1, 2), (2, 7), (3, 8), (4, 6), (5, 5)]);

        t1.set_bond_dims(&bond_dims);
        t2.set_bond_dims(&bond_dims);
        t3.set_bond_dims(&bond_dims);

        tout.set_bond_dims(&bond_dims);
        let edges = tout.get_mut_edges();
        edges
            .entry(5)
            .or_insert(vec![Vertex::Closed(0), Vertex::Open]);
        edges
            .entry(4)
            .or_insert(vec![Vertex::Closed(0), Vertex::Open]);
        edges
            .entry(1)
            .or_insert(vec![Vertex::Closed(0), Vertex::Open]);
        edges.entry(2).or_insert(vec![Vertex::Closed(0)]);
        edges.entry(3).or_insert(vec![Vertex::Closed(0)]);

        let (d1, d2, d3, dout) = setup();

        t1.set_tensor_data(TensorData::new_from_data(
            t1.shape(),
            d1,
            Some(Layout::RowMajor),
        ));

        t2.set_tensor_data(TensorData::new_from_data(
            t2.shape(),
            d2,
            Some(Layout::RowMajor),
        ));
        t3.set_tensor_data(TensorData::new_from_data(
            t3.shape(),
            d3,
            Some(Layout::RowMajor),
        ));
        tout.set_tensor_data(TensorData::new_from_data(
            tout.shape(),
            dout,
            Some(Layout::RowMajor),
        ));

        let mut tn = create_tensor_network(vec![t1, t2, t3], &bond_dims, None);
        let contract_path = path![(0, 1), (0, 2)];

        contract_tensor_network(&mut tn, &contract_path);

        assert_eq!(tout, tn);
    }

    //     #[test]
    //     fn test_tn_partitioned_contraction() {
    //         // t1 is of shape [3, 2, 7]
    //         let t1 = Tensor::new(vec![0, 1, 2]);
    //         // t3 is of shape [7, 8, 6]
    //         let t3 = Tensor::new(vec![2, 3, 4]);
    //         // t5 is of shape [3, 5, 8]
    //         let t5 = Tensor::new(vec![0, 5, 3]);

    //         // t2 is of shape [6, 2, 4]
    //         let t2 = Tensor::new(vec![4, 6, 7]);
    //         // t4 is of shape [5, 2]
    //         let t4 = Tensor::new(vec![5, 6]);

    //         // tout is of shape [4, 2]
    //         let tout = Tensor::new(vec![7, 1]);

    //         // let (d1, d2, d3, d4, d5, dout) = setup();

    //         let bond_dims = HashMap::from([
    //             (0, 3),
    //             (1, 2),
    //             (2, 7),
    //             (3, 8),
    //             (4, 6),
    //             (5, 5),
    //             (6, 2),
    //             (7, 4),
    //             (8, 3),
    //             (9, 4),
    //             (10, 2),
    //             (11, 4),
    //             (12, 3),
    //         ]);

    //         t1.set_tensor_data(TensorData::new_from_flat(
    //             t1.shape(),
    //             d1,
    //             Some(tetra::Layout::RowMajor),
    //         ));
    //         t2.set_tensor_data(TensorData::new_from_flat(
    //             t2.shape(),
    //             d2,
    //             Some(tetra::Layout::RowMajor),
    //         ));
    //         t3.set_tensor_data(TensorData::new_from_flat(
    //             t3.shape(),
    //             d3,
    //             Some(tetra::Layout::RowMajor),
    //         ));
    //         t4.set_tensor_data(TensorData::new_from_flat(
    //             t4.shape(),
    //             d4,
    //             Some(tetra::Layout::RowMajor),
    //         ));
    //         t5.set_tensor_data(TensorData::new_from_flat(
    //             t5.shape(),
    //             d5,
    //             Some(tetra::Layout::RowMajor),
    //         ));
    //         tout.set_tensor_data(TensorData::new_from_flat(
    //             tout.shape(),
    //             dout,
    //             Some(tetra::Layout::RowMajor),
    //         ));

    //         let tn = create_tensor_network(vec![t1, t2, t3, t4, t5], &bond_dims, None);
    //         let mut partitioned_tn = partition_tensor_network(&tn, &[1, 0, 1, 0, 1]);
    //         let contract_path = vec![
    //             ContractionIndex::Path((0, path![(0, 1)])),
    //             ContractionIndex::Path((1, path![(0, 1), (0, 2)])),
    //             (0, 1).into(),
    //         ];
    //         contract_tensor_network(&mut partitioned_tn, &contract_path);
    //         let result = tn.get_data();
    //         let ref_result = tout.get_data();

    //         let range = tout
    //             .shape()
    //             .iter()
    //             .map(|e| 0..*e as u32)
    //             .multi_cartesian_product();

    //         for index in range {
    //             assert!(approx_eq!(
    //                 f64,
    //                 ref_result.get(index.as_slice()).re,
    //                 result.get(index.as_slice()).re,
    //                 epsilon = 1e-8
    //             ));
    //             assert!(approx_eq!(
    //                 f64,
    //                 ref_result.get(index.as_slice()).im,
    //                 result.get(index.as_slice()).im,
    //                 epsilon = 1e-8
    //             ));
    //         }
    //     }
}
