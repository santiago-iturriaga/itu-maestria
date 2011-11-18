/*
  Copyright (c) 2007 A. Arnold and J. A. van Meel, FOM institute
  AMOLF, Amsterdam; all rights reserved unless otherwise stated.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  In addition to the regulations of the GNU General Public License,
  publications and communications based in parts on this program or on
  parts of this program are required to cite the article
  "Harvesting graphics power for MD simulations"
  by J.A. van Meel, A. Arnold, D. Frenkel, S. F. Portegies Zwart and
  R. G. Belleman, Molecular Simulation, Vol. 34, p. 259 (2007).

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA 02111-1307 USA
*/
#ifndef RANDOM_H_
#define RANDOM_H_

/** a rand48 random number generator.
    The random number generator works similar to the standard lrand48.
    The seed, which is normally set by srand48, here a parameter of the constructor.
    Random numbers are drawn in two steps:
    - first, random numbers are generated via the generate function, and then
    - then, they can be retrieved via the get function
    Alternatively, you can use them directly on the GPU. A pointer to the random
    number array can be retrieved by get_random_numbers. This functions returns a
    void *, to avoid CUDA data types in this header file. The true type
    is however int *.
*/

struct RNG_rand48 {
  int stride;

  /// current random numbers per thread
  void *state;
  
  /// generated random numbers
  void *res;

  /// number of threads in each block
  int threadsX;
  
  /// number of blocks of threads
  int blocksX;

  /// random numbers to generate
  int rand_num_count;

  /// total number of threads
  int nThreads;
  
  /// random numbers to generate per thread
  int num_blocks;

  /** strided iteration constants (48-bit, distributed on 2x 24-bit) */
  unsigned int A0, A1, C0, C1;

  /// magic constants for rand48
  unsigned long long a, c;
}; 

/// CUDA-safe constructor
void RNG_rand48_init(struct RNG_rand48 &rand_state, int seed, int count);

/// CUDA-safe destructor
void RNG_rand48_cleanup(struct RNG_rand48 &rand_state);

/// generate n random numbers as 31-bit integers like lrand48
void RNG_rand48_generate(struct RNG_rand48 &rand_state);

/** get the first n of the previously generated numbers into array r.
  r must be large enough to contain all the numbers, and enough
  numbers have to be generated before. */
void RNG_rand48_get(struct RNG_rand48 &rand_state, int *r);

#endif
