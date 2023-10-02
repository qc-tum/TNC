use itertools::izip;
use itertools::Itertools;

use super::contraction::tn_output_tensor;
use super::DataTensor;
use super::{contraction::tn_contract, tensor::Tensor, TensorNetwork};
use tetra::permutation::Permutation;

/// Stores feynman contraction data specific to a given feynman scattering.
pub struct FeynmanContractionData {
    /// Vector of usize indicating the edge ids of feynman indices
    pub feynman_indices: Vec<usize>,
    /// Stores the permutation required to ensure each Tensor has feynman indices as slowest running index
    pub permutation_vector: Vec<Permutation>,
    /// Stores the index identifying which feynman index is iterated over for each Tensor object
    pub feynman_tensor_indexes: Vec<Vec<usize>>,
}

/// Slices a [`Tensor`] along given `feynman_indices`.
///
/// # Arguments
///
/// * `t` - [`&Tensor`] to be sliced
/// * `feynman_indices` - &[usize] containing feynman indices in tensor network
///
fn scatter_tensor(t: &Tensor, feynman_indices: &[usize]) -> (Tensor, Permutation, Vec<usize>) {
    let mut new_legs = Vec::new();
    let mut perm = Vec::new();
    let mut feynman_perm = Vec::new();
    let mut feynman_indexing = Vec::new();

    for (i, index) in t.get_legs().iter().enumerate() {
        if !feynman_indices.contains(index) {
            new_legs.push(*index);
            perm.push(i);
        } else {
            feynman_perm.push(i);
            feynman_indexing.push(feynman_indices.iter().position(|&r| r == *index).unwrap())
        }
    }
    perm.append(&mut feynman_perm);
    (
        Tensor::new(new_legs),
        Permutation::between(&(0..t.get_legs().len()).collect_vec(), &perm),
        feynman_indexing,
    )
}

/// Slices all [`Tensor`] objects in a [`TensorNetwork`] along given `feynman_indices`.
/// Returns a new ['TensorNetwork'] without any `feynman_indices` but retains sliced edge information
/// in `ext_edges`, `edges` and `bond_dims`; a `Vector<tetra::Permutation>` that indicates the required permutation
/// to set feynman indices as the slowest running index; and a `Vec<Vec<usize>>` indicating the corresponding
/// feynman index (if one exists) for each ['Tensor'] in the output [`TensorNetwork`].
///
/// # Arguments
///
/// * `tn` - [`&TensorNetwork`] to be sliced
/// * `feynman_indices` - &[usize] containing feynman indices in tensor network
///
///
/// # Examples
///
/// ```
/// # extern crate tensorcontraction;
/// # use tensorcontraction::{
///     contractionpath::paths::{BranchBound, CostType, OptimizePath},
///     random::tensorgeneration::{random_sparse_tensor, random_tensor_network},
///     tensornetwork::{tensor::Tensor,
///     TensorNetwork,
///     contraction::tn_contract,
///     feynman::{feynman_scatter, FeynmanContractionData}}
/// };
/// use std::collections::HashMap;
/// use tetra::permutation::Permutation;
/// let t1 = Tensor::new(vec![0, 1, 2]);
/// let t2 = Tensor::new(vec![2, 3, 4]);
/// let bond_dims = HashMap::from([(0, 3), (1, 2), (2, 7), (3, 8), (4, 6)]);
/// let tn = TensorNetwork::new(vec![t1, t2], bond_dims.clone(), None);
/// let feynman_indices = &[0, 3];
/// let (feyn_tn, feynman_options) = feynman_scatter(&tn, feynman_indices);
/// let FeynmanContractionData{    
/// feynman_indices,
/// permutation_vector,
/// feynman_tensor_indexes,
/// } = feynman_options;
/// assert_eq!(permutation_vector, vec![Permutation::new(vec![1, 2, 0]), Permutation::new(vec![0, 2, 1])]);
/// assert_eq!(feynman_tensor_indexes, vec![[0], [1]]);
/// assert_eq!(*feyn_tn.get_tensors(), vec![Tensor::new(vec![1, 2]), Tensor::new(vec![2, 4])]);
/// assert_eq!(*feyn_tn.get_bond_dims(), bond_dims);
/// ```
pub fn feynman_scatter(
    tn: &TensorNetwork,
    feynman_indices: &[usize],
) -> (TensorNetwork, FeynmanContractionData) {
    let mut feynman_tensors = Vec::with_capacity(tn.get_tensors().len());
    let mut permutation_vector = Vec::with_capacity(tn.get_tensors().len());
    let mut feynman_tensor_indexes = Vec::with_capacity(tn.get_tensors().len());
    for tensor in tn.get_tensors() {
        let (sliced_tensor, perm, feynman_index) = scatter_tensor(tensor, feynman_indices);
        feynman_tensor_indexes.push(feynman_index);
        feynman_tensors.push(sliced_tensor);
        permutation_vector.push(perm);
    }

    (
        TensorNetwork {
            tensors: feynman_tensors,
            bond_dims: tn.get_bond_dims().clone(),
            edges: tn.get_edges().clone(),
            ext_edges: tn.get_ext_edges().clone(),
        },
        FeynmanContractionData {
            feynman_indices: feynman_indices.to_vec(),
            permutation_vector,
            feynman_tensor_indexes,
        },
    )
}

/// Calculates the size of contiguous memory of a given ['DataTensor']
///
/// # Arguments
///
/// * `dt` - [`&DataTensor`] to be sliced
/// * `tensor_len` - number of legs in [`&DataTensor`]
///

fn calculate_chunk_size(dt: &DataTensor, tensor_len: usize) -> usize {
    let c_chunk_size = dt.shape().iter().copied().take(tensor_len).product::<u32>() as usize;
    c_chunk_size
}

/// Calculates the feynman index for a given ['DataTensor']
///
/// # Arguments
///
/// * `dt` - [`&DataTensor`] to be sliced
/// * `feynman_index` - &[usize] containing feynman indices in tensor network
///
fn calculate_feynman_index(dt: &DataTensor, feynman_index: &Vec<u32>) -> usize {
    let index_value = dt
        .shape()
        .iter()
        .rev()
        .take(feynman_index.len())
        .zip(feynman_index.iter())
        .rev()
        .fold(0, |current_index, (dim, index)| {
            current_index * (*dim as usize) + (*index as usize)
        });
    index_value
}

/// Slices a [`DataTensor`] along given `feynman_indices`, returning values associated with the provided feynmna index.
/// Assumes that passed `DataTensor` is already permuted such that the sliced indices are now the slowest running index.
///
/// # Arguments
///
/// * `dt` - [`&DataTensor`] to be sliced
/// * `feynman_indices` - &[usize] containing feynman indices in tensor network
///
fn feynman_slice_data_tensor(dt: &DataTensor, feynman_index: &Vec<u32>) -> DataTensor {
    assert!(feynman_index.len() < dt.ndim());
    let tensor_len = dt.ndim() - feynman_index.len();
    let c_chunk_size = calculate_chunk_size(dt, tensor_len);
    let index_value = calculate_feynman_index(dt, feynman_index);
    DataTensor::new_from_flat(
        &dt.shape()[0..tensor_len],
        dt.get_raw_data()[index_value * c_chunk_size..(index_value + 1) * c_chunk_size].to_vec(),
        None,
    )
}

/// Inserts values of `dt_src` into a slice of `dt_dest` where the feynman indices are fixed to the given
/// `feynman_index` values. Assumes that `dt_src` is the exact size of the sliced `dt_dest`
/// Also assumes that passed `DataTensor` is already permuted such that the feynman indices are
/// the slowest running, which means the replaced data is contiguous.
///
/// # Arguments
///
/// * `dt_dest` - destination `&mut DataTensor`
/// * `feynman_indices` - &[usize] containing feynman indices in tensor network
/// * `dt_src` - source &DataTensor
///
fn feynman_insert_data_tensor(
    dt_dest: &mut DataTensor,
    feynman_index: &Vec<u32>,
    dt_src: &DataTensor,
) {
    assert!(feynman_index.len() < dt_dest.ndim());
    let tensor_len: usize = dt_dest.ndim() - feynman_index.len();
    let c_chunk_size = calculate_chunk_size(dt_dest, tensor_len);
    let index_value = calculate_feynman_index(dt_dest, feynman_index);
    assert_eq!(c_chunk_size as u32, dt_src.size(None));
    dt_dest.get_raw_data_mut().splice(
        index_value * c_chunk_size..(index_value + 1) * c_chunk_size,
        dt_src.get_raw_data().iter().copied(),
    );
}

/// Performs a feynamn contraction on a given scattered `TensorNetwork`
///
/// # Arguments
///
/// * `feynman_tn` - [`TensorNetwork`] that has been scattered via `feynman_scatter`
/// * `d_tn` - mut `Vec<DataTensor>` containing raw tensor data
/// * `contract_path` - `Vector` of `(usize, usize)`, indicating contraction path. See [`BranchBound`] for details on `contract_path` format.
/// * `out_indices` - `&[usize]` specifying output shape of contracted `TensorNetwork`
/// * `feynman_options` - [`FeynmanContractionData`] object, output of `feynman_scatter` function
///
pub fn feynman_contraction(
    feynman_tn: TensorNetwork,
    mut d_tn: Vec<DataTensor>,
    contract_path: &Vec<(usize, usize)>,
    out_indices: &[usize],
    feynman_options: FeynmanContractionData,
) -> DataTensor {
    let FeynmanContractionData {
        feynman_indices,
        permutation_vector,
        feynman_tensor_indexes,
    } = feynman_options;

    let bond_dims = feynman_tn.get_bond_dims();

    let feynman_index_sizes = feynman_indices
        .iter()
        .map(|e| *bond_dims.get(e).unwrap())
        .collect::<Vec<u64>>();

    for (d_t, perm) in d_tn.iter_mut().zip(permutation_vector) {
        d_t.transpose(&perm);
    }

    let mut feynman_output = tn_output_tensor(feynman_tn.clone(), contract_path);

    feynman_output.append(&mut feynman_indices.to_vec());

    let output_shape = feynman_output
        .iter()
        .map(|e| bond_dims[e] as u32)
        .collect::<Vec<u32>>();

    let mut out_tensor = DataTensor::new(&output_shape);

    let feynman_range = feynman_index_sizes
        .iter()
        .map(|e| 0..*e as u32)
        .multi_cartesian_product();

    for index in feynman_range {
        let mut d_tn_sliced = Vec::with_capacity(d_tn.len());
        for (d_t, sliced_index) in izip!(&d_tn, &feynman_tensor_indexes) {
            let sliced_values = sliced_index.iter().map(|&e| index[e]).collect_vec();
            let sliced_tensor = feynman_slice_data_tensor(d_t, &sliced_values);
            d_tn_sliced.push(sliced_tensor);
        }
        let (_, d_out) = tn_contract(feynman_tn.clone(), d_tn_sliced, contract_path);
        feynman_insert_data_tensor(&mut out_tensor, &index, &d_out[0]);
    }

    let out_perm = Permutation::between(&feynman_output, out_indices);
    out_tensor.transpose(&out_perm);
    out_tensor
}

#[cfg(test)]
mod tests {
    use super::{feynman_contraction, feynman_scatter, FeynmanContractionData};
    use crate::tensornetwork::{tensor::Tensor, TensorNetwork};
    use float_cmp::{approx_eq, assert_approx_eq};
    use itertools::Itertools;
    use num_complex::Complex64;
    use std::collections::HashMap;
    use tetra::{permutation::Permutation, Tensor as DataTensor};

    fn row_major_setup() -> (
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

    fn col_major_setup() -> (
        Vec<Complex64>,
        Vec<Complex64>,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let d1 = vec![
            Complex64::new(0.17560865095087674, 0.8387771044087811),
            Complex64::new(0.4020080006121739, 0.19205211684777945),
            Complex64::new(0.565812341997073, 0.2054129028944197),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.1881030044755968, 0.6001727959338155),
            Complex64::new(0.35567185231766396, 0.8440720998918102),
            Complex64::new(0.18461234934069493, 0.020129617529815902),
            Complex64::new(0.20843326736337642, 0.12277457339753695),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.4037636828939306, 0.017423147124870697),
            Complex64::new(0.20203697022026124, 0.01275349965390471),
            Complex64::new(0.6515506736925029, 0.4924644609220955),
            Complex64::new(0.7567346462709171, 0.42828065225643097),
            Complex64::new(0.0, 0.0),
        ];

        let d2 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.9359910317970342, 0.34974572309035323),
            Complex64::new(0.42711288030965144, 0.7816795439302059),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.9182848935433582, 0.6901786922879987),
            Complex64::new(0.7716316920912436, 0.5776373383707031),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.9144011577505171, 0.728132842331050),
        ]
        .to_vec();

        let d3 = vec![
            Complex64::new(0.06341055580697552, 0.2589720976240343),
            Complex64::new(0.7782666939493721, 0.7092562072042309),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.48960291550214097, 0.9945540993278725),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.32281503514932663, 0.9400547654920804),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5677370469313697, 0.050389392788741),
        ]
        .to_vec();

        let dout = vec![
            Complex64::new(-3.2324764062282636, 1.5205693623709797),
            Complex64::new(-1.5575397714234704, 1.210391237786501),
            Complex64::new(-0.6515228792056469, 1.3312524247584296),
            Complex64::new(0.0022782709134253065, 0.9703268064501953),
        ]
        .to_vec();
        (d1, d2, d3, dout)
    }

    #[test]
    fn test_feynman_scatter() {
        let t1 = Tensor::new(vec![0, 1, 2]);
        // t2 is of shape [7, 8, 6]
        let t2 = Tensor::new(vec![2, 3, 4]);
        // t3 is of shape [3, 5, 8]
        let t3 = Tensor::new(vec![0, 5, 3]);
        // tout is of shape [5, 6, 2]
        // let tout = Tensor::new(vec![3, 4, 0, 1]);

        let bond_dims = HashMap::from([(0, 3), (1, 2), (2, 7), (3, 8), (4, 6), (5, 5)]);

        let tn = TensorNetwork::new(vec![t1, t2, t3], bond_dims, None);

        let feynman_indices = [4, 1, 5];

        let (feynman_tn, feynman_options) = feynman_scatter(&tn, &feynman_indices);

        let feynman_tensor_ref = vec![
            Tensor::new(vec![0, 2]),
            Tensor::new(vec![2, 3]),
            Tensor::new(vec![0, 3]),
        ];
        let FeynmanContractionData {
            feynman_indices: _,
            permutation_vector,
            feynman_tensor_indexes,
        } = feynman_options;

        for (i, tensor) in feynman_tn.get_tensors().iter().enumerate() {
            assert_eq!(tensor.get_legs(), feynman_tensor_ref[i].get_legs());
        }
        let perm_vector_ref = vec![
            Permutation::new(vec![0, 2, 1]),
            Permutation::new(vec![0, 1, 2]),
            Permutation::new(vec![0, 2, 1]),
        ];

        for (i, perm) in permutation_vector.iter().enumerate() {
            assert_eq!(&perm_vector_ref[i], perm);
        }

        let feynman_index_ref = vec![[1], [0], [2]];

        for (i, feynman_index) in feynman_tensor_indexes.iter().enumerate() {
            assert_eq!(&feynman_index_ref[i].to_vec(), feynman_index);
        }
    }

    #[test]
    fn test_simple_feynman() {
        let solution_data = vec![
            Complex64::new(1.1913917228026232, -3.7863595014806157),
            Complex64::new(1.5884274662744466, 1.1478771890194843),
        ];
        let b_data = vec![
            Complex64::new(1.764052345967664, -0.10321885179355784),
            Complex64::new(1.8675579901499675, 0.7610377251469934),
            Complex64::new(0.9787379841057392, 0.144043571160878),
            Complex64::new(0.9500884175255894, 0.44386323274542566),
            Complex64::new(0.4001572083672233, 0.41059850193837233),
            Complex64::new(-0.977277879876411, 0.12167501649282841),
            Complex64::new(2.240893199201458, 1.454273506962975),
            Complex64::new(-0.1513572082976979, 0.33367432737426683),
        ];
        let c_data = vec![
            Complex64::new(1.4940790731576061, -2.5529898158340787),
            Complex64::new(0.31306770165090136, 0.8644361988595057),
            Complex64::new(-0.20515826376580087, 0.6536185954403606),
            Complex64::new(-0.8540957393017248, -0.7421650204064419),
        ];

        let tc1 = DataTensor::new_from_flat(&[2, 2, 2], b_data, None);
        let tc2 = DataTensor::new_from_flat(&[2, 2, 1], c_data, None);
        let tcout = DataTensor::new_from_flat(&[2, 1], solution_data, None);

        let t1 = Tensor::new(vec![1, 0, 2]);
        let t2 = Tensor::new(vec![0, 1, 3]);
        // let tout = Tensor::new(vec![2, 3]);
        let bond_dims = HashMap::from([(0, 2), (1, 2), (2, 2), (3, 1)]);

        let feynman_indices = vec![2];

        let tn = TensorNetwork::new(vec![t1, t2], bond_dims, None);

        let (feynman_tn, feynman_options) = feynman_scatter(&tn, &feynman_indices);

        let contract_path = vec![(0, 1)];

        let d_t = feynman_contraction(
            feynman_tn,
            vec![tc1, tc2],
            &contract_path,
            &[2, 3],
            feynman_options,
        );

        let range = d_t.shape().iter().map(|e| 0..*e).multi_cartesian_product();
        for index in range {
            assert!(approx_eq!(
                f64,
                tcout.get(&index).re,
                d_t.get(&index).re,
                epsilon = 1e-8
            ));
            assert!(approx_eq!(
                f64,
                tcout.get(&index).im,
                d_t.get(&index).im,
                epsilon = 1e-8
            ));
        }
    }

    #[test]
    fn test_col_major_feynman() {
        let t1 = Tensor::new(vec![0, 1, 2, 3]);
        let t2 = Tensor::new(vec![0, 1, 4]);
        let t3 = Tensor::new(vec![4, 2, 5]);
        // let tout = Tensor::new(vec![3, 5]);
        let bond_dims = HashMap::from([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]);

        let (d1, d2, d3, dout) = col_major_setup();

        let dt1 = DataTensor::new_from_flat(&[2, 2, 2, 2], d1, None);
        let dt2 = DataTensor::new_from_flat(&[2, 2, 2], d2, None);
        let dt3 = DataTensor::new_from_flat(&[2, 2, 2], d3, None);
        let dout = DataTensor::new_from_flat(&[2, 2], dout, None);

        let opt_path = vec![(0, 1), (0, 2)];

        let tn = TensorNetwork::new(vec![t1, t2, t3], bond_dims, None);

        let feynman_indices = [5];

        let (feynman_tn, feynman_options) = feynman_scatter(&tn, &feynman_indices);

        let d_t = feynman_contraction(
            feynman_tn,
            vec![dt1, dt2, dt3],
            &opt_path,
            &[5, 3],
            feynman_options,
        );

        let range = d_t.shape().iter().map(|e| 0..*e).multi_cartesian_product();
        for index in range {
            assert!(approx_eq!(
                f64,
                dout.get(&index).re,
                d_t.get(&index).re,
                epsilon = 1e-8
            ));
            assert!(approx_eq!(
                f64,
                dout.get(&index).im,
                d_t.get(&index).im,
                epsilon = 1e-8
            ));
        }
    }

    #[test]
    fn test_row_major_contraction() {
        // t1 is of shape [3, 2, 7]
        let t1 = Tensor::new(vec![0, 1, 2]);
        // t2 is of shape [7, 8, 6]
        let t2 = Tensor::new(vec![2, 3, 4]);
        // t3 is of shape [3, 5, 8]
        let t3 = Tensor::new(vec![0, 5, 3]);
        // tout is of shape [5, 6, 2]
        let tout = Tensor::new(vec![5, 4, 1]);
        // let tout = Tensor::new(vec![3, 4, 0, 1]);

        let (d1, d2, d3, dout) = row_major_setup();

        let bond_dims = HashMap::from([(0, 3), (1, 2), (2, 7), (3, 8), (4, 6), (5, 5)]);

        let tc1 = DataTensor::new_from_flat(
            &(t1.iter().map(|e| bond_dims[e] as u32).collect::<Vec<u32>>()),
            d1,
            Some(tetra::Layout::RowMajor),
        );
        let tc2 = DataTensor::new_from_flat(
            &(t2.iter().map(|e| bond_dims[e] as u32).collect::<Vec<u32>>()),
            d2,
            Some(tetra::Layout::RowMajor),
        );
        let tc3 = DataTensor::new_from_flat(
            &(t3.iter().map(|e| bond_dims[e] as u32).collect::<Vec<u32>>()),
            d3,
            Some(tetra::Layout::RowMajor),
        );
        let tcout = DataTensor::new_from_flat(
            &(tout
                .iter()
                .map(|e| bond_dims[e] as u32)
                .collect::<Vec<u32>>()),
            dout,
            Some(tetra::Layout::RowMajor),
        );

        let tn = TensorNetwork::new(vec![t1, t2, t3], bond_dims, None);
        let contract_path = vec![(0, 1), (0, 2)];

        let feynman_indices = [1];

        let (feynman_tn, feynman_options) = feynman_scatter(&tn, &feynman_indices);

        let d_t = feynman_contraction(
            feynman_tn,
            vec![tc1, tc2, tc3],
            &contract_path,
            &[5, 4, 1],
            feynman_options,
        );

        let range = d_t.shape().iter().map(|e| 0..*e).multi_cartesian_product();

        for index in range {
            assert_approx_eq!(
                f64,
                tcout.get(&index).re,
                d_t.get(&index).re,
                epsilon = 1e-8
            );
            assert_approx_eq!(
                f64,
                tcout.get(&index).im,
                d_t.get(&index).im,
                epsilon = 1e-8
            );
        }
    }
}
