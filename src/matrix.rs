#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

pub fn make_matrix(rows: usize, cols: usize) -> Matrix {
    let mut data = vec![];
    for _ in 0..rows {
        data.push(vec![0.0; cols as usize]);
    }
    return Matrix { rows, cols, data };
}

pub fn copy_matrix(m: &Matrix) -> Matrix {
    let mut n = make_matrix(m.rows, m.cols);
    for x in 0..m.cols {
        for y in 0..m.rows {
            n.data[x][y] = m.data[x][y];
        }
    }
    return n;
}

pub fn make_identity_homography() -> Matrix {
    let mut h = make_matrix(3, 3);
    h.data[0][0] = 1.0;
    h.data[1][1] = 1.0;
    h.data[2][2] = 1.0;
    return h;
}

pub fn make_translation_homography(dx: f64, dy: f64) -> Matrix {
    let mut h = make_identity_homography();
    h.data[0][2] = dx;
    h.data[1][2] = dy;
    return h;
}

pub fn augment_matrix(m: &Matrix) -> Matrix {
    let mut c = make_matrix(m.rows, m.cols * 2);
    for y in 0..m.cols as usize {
        for x in 0..m.rows as usize {
            c.data[x][y] = m.data[x][y];
        }
    }
    for x in 0..m.rows as usize {
        c.data[x][x + m.cols as usize] = 1.0;
    }
    return c;
}

pub fn make_identity(rows: usize, cols: usize) -> Matrix {
    let mut m = make_matrix(rows, cols);
    for i in 0..rows as usize {
        m.data[i][i] = 1.0;
    }
    return m;
}

pub fn matrix_mult_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let mut p = make_matrix(a.rows, b.cols);
    for i in 0..p.rows as usize {
        for j in 0..p.cols as usize {
            for k in 0..a.cols as usize {
                p.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return p;
}

pub fn matrix_elmult_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.rows, b.rows);
    let mut p = make_matrix(a.rows, a.cols);
    for i in 0..p.rows as usize {
        for j in 0..p.cols as usize {
            p.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return p;
}

pub fn matrix_sub_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.rows, b.rows);
    let mut p = make_matrix(a.rows, a.cols);
    for i in 0..p.rows as usize {
        for j in 0..p.cols as usize {
            p.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return p;
}

pub fn axpy_matrix(a: f64, x: &Matrix, y: &Matrix) -> Matrix {
    assert_eq!(x.cols, y.cols);
    assert_eq!(x.rows, y.rows);
    let mut p = make_matrix(x.rows, x.cols);
    for i in 0..x.rows as usize {
        for j in 0..x.cols as usize {
            p.data[i][j] = a * x.data[i][j] + y.data[i][j];
        }
    }
    return p;
}

pub fn transpose_matrix(m: &Matrix) -> Matrix {
    let rows = m.cols;
    let cols = m.rows;
    let mut data = vec![vec![0.0; cols as usize]; rows as usize];
    for i in 0..rows as usize {
        for j in 0..cols as usize {
            data[i][j] = data[j][i];
        }
    }
    return Matrix { rows, cols, data };
}

pub fn scale_matrix(m: &mut Matrix, s: f64) {
    for i in 0..m.rows as usize {
        for j in 0..m.cols as usize {
            m.data[i][j] *= s;
        }
    }
}

pub fn matrix_invert(m: &Matrix) -> Option<Matrix> {
    if m.rows != m.cols {
        return None;
    }

    let mut identity = make_identity(m.rows, m.cols);
    let mut copy = copy_matrix(&m);

    for i in 0..copy.rows {
        let mut e = copy.data[i][i];

        if e == 0.0 {
            for ii in (i + 1)..copy.rows {
                if copy.data[ii][i] != 0.0 {
                    for j in 0..copy.rows {
                        e = copy.data[i][j];
                        copy.data[i][j] = copy.data[ii][j];
                        copy.data[ii][j] = e;
                        e = identity.data[i][j];
                        identity.data[i][j] = identity.data[ii][j];
                        identity.data[ii][j] = e;
                    }
                    break;
                }
            }

            e = copy.data[i][i];
            if e == 0.0 {
                return None;
            }
        }

        for j in 0..copy.rows {
            copy.data[i][j] = copy.data[i][j] / e;
            identity.data[i][j] = identity.data[i][j] / e;
        }

        for ii in 0..copy.rows {
            if ii == i {
                continue;
            }
            e = copy.data[ii][i];
            for j in 0..copy.rows {
                copy.data[ii][j] -= e * copy.data[i][j];
                identity.data[ii][j] -= e * identity.data[i][j];
            }
        }
    }

    return Some(identity);
}

// pub fn matrix_invert(m: &Matrix) -> Option<Matrix> {
//     if m.rows != m.cols {
//         return None;
//     }
//     let mut c = augment_matrix(m);
//     for k in 0..c.rows {
//         let mut p = 0.0;
//         let mut index = -1;
//         for i in k..c.rows {
//             let val = c.data[i][k].abs();
//             if val > p {
//                 p = val;
//                 index = i as i32;
//             }
//         }
//         println!("{}", index);
//         if index == -1 {
//             return None;
//         }

//         let swap = c.data[index as usize].clone();
//         c.data[index as usize] = c.data[k].clone();
//         c.data[k] = swap;

//         let val = c.data[k][k];
//         c.data[k][k] = 1.0;
//         if k < c.cols - 1 {
//             println!("col count {}", c.cols);
//             println!("{}", k);
//             for j in (k + 1)..c.cols {
//                 println!("cols {}", j);
//                 c.data[k][j] /= val;
//             }
//         }
//         if k < c.rows - 1 {
//             println!("row count {}", c.rows);
//             for i in (k + 1)..c.rows {
//                 println!("rows {}", i);
//                 let s = -c.data[i][k];
//                 c.data[i][k] = 0.0;
//                 for j in (k + 1)..c.cols {
//                     c.data[i][j] += s * c.data[k][j];
//                 }
//             }
//         }
//     }

//     for k in (c.rows - 1)..0 {
//         for i in 0..k {
//             let s = -c.data[i][k];
//             c.data[i][k] = 0.0;
//             for j in (k + 1)..c.cols {
//                 c.data[i][j] += s * c.data[k][j];
//             }
//         }
//     }

//     let mut inv = make_matrix(m.rows, m.cols);
//     for i in 0..m.rows {
//         for j in 0..m.cols {
//             inv.data[i][j] = c.data[i][j + m.cols];
//         }
//     }

//     return Some(inv);
// }

pub fn solve_system(m: &Matrix, b: &Matrix) -> Option<Matrix> {
    let mt = transpose_matrix(m);
    let mtm = matrix_mult_matrix(&mt, m);
    let opt_mtm_inv = matrix_invert(&mtm);
    match opt_mtm_inv {
        Some(mtm_inv) => {
            let mdag = matrix_mult_matrix(&mtm_inv, &mt);
            let a = matrix_mult_matrix(&mdag, b);
            return Some(a);
        }
        None => {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_matrix() {
        let mut a = make_matrix(2, 2);
        a.data[0] = vec![4.0, 7.0];
        a.data[1] = vec![2.0, 6.0];

        if let Some(inv) = matrix_invert(&a) {
            assert_eq!(inv.data[0][0], 0.6);
            assert_eq!(inv.data[0][1], -0.7);
            assert_eq!(inv.data[1][0], -0.2);
            assert_eq!(inv.data[1][1], 0.4);
        } else {
            assert!(false);
        }
    }
}
