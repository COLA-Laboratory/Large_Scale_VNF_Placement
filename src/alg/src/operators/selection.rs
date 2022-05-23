use rand::{thread_rng, Rng};
pub struct TournamentSelection<T>
where
    T: Fn(usize, usize) -> bool,
{
    beats: T,
    pop_size: usize,
}

impl<T> TournamentSelection<T>
where
    T: Fn(usize, usize) -> bool,
{
    pub fn new(pop_size: usize, beats: T) -> Self {
        TournamentSelection { beats, pop_size }
    }

    pub fn tournament(&self, tournament_size: usize) -> usize {
        if tournament_size == 0 {
            panic!("Tournament size must be 1 or greater");
        }

        let mut rng = thread_rng();
        let mut used = Vec::new();

        let mut curr_best = rng.gen_range(0..self.pop_size);

        used.push(curr_best);

        for _ in 0..tournament_size - 1 {
            let mut contender;
            loop {
                contender = rng.gen_range(0..self.pop_size);

                if !used.contains(&contender) {
                    used.push(contender);
                    break;
                }
            }

            if (self.beats)(contender, curr_best) {
                curr_best = contender;
            }
        }

        curr_best
    }
}
