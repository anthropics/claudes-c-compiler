pub fn is_multiple_of<T: std::ops::Rem<Output = T> + PartialEq + From<u8> + Copy>(value: T, divisor: T) -> bool {
    divisor != T::from(0) && value % divisor == T::from(0)
}
