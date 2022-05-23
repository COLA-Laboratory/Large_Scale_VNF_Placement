pub fn round_to(num: f64, num_dp: usize) -> f64 {
    let ten: f64 = 10.0;
    let mult = ten.powf(num_dp as f64);
    (num * mult).round() / mult
}

pub fn normalize_in_place(vectors: &mut Vec<Vec<f64>>) {
    let d = vectors[0].len();

    let mut min = vec![std::f64::MAX; d];
    let mut max = vec![std::f64::MIN; d];

    for vect in vectors.iter() {
        for i in 0..d {
            if vect[i] < min[i] {
                min[i] = vect[i];
            }

            if vect[i] > max[i] {
                max[i] = vect[i];
            }
        }
    }

    for i in 0..vectors.len() {
        for j in 0..d {
            vectors[i][j] = (vectors[i][j] - min[j]) / (max[j] - min[j]);
        }
    }

    // TODO: Write unit tests
}

pub fn euclidean_distance(vec_i: &Vec<f64>, vec_j: &Vec<f64>) -> f64 {
    let mut sum = 0.0; 

    for i in 0..vec_i.len() {
        sum += (vec_i[i] - vec_j[i]).powf(2.0); 
    }

    sum.sqrt()
}

// ----- Unit tests ---- //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_round_to() {
        let pi = std::f64::consts::PI;

        let zero = round_to(pi, 0);
        let one = round_to(pi, 1);
        let two = round_to(pi, 2);
        let three = round_to(pi, 3);
        let four = round_to(pi, 4);

        assert_eq!(zero, 3.0); // (Pi is exactly 3!)
        assert_eq!(one, 3.1);
        assert_eq!(two, 3.14);
        assert_eq!(three, 3.142);
        assert_eq!(four, 3.1416);
    }
}
