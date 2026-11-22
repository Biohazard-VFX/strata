# Architecture decisions

- Score conditional alpha in fixed slot 0 as the foreground proxy; explicit GT-independent membership uses sum(m_k*a_k*prod(j<k,1-a_j)), never total scene opacity or oracle layer selection.
