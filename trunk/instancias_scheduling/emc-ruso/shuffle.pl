#!/usr/bin/perl -w

# shuffle a file

# Set the random seed
srand(time|$$);

# Suck in everything in the file.
@a = <>;

# Get random lines, write 'em out, mark 'em done.
while ( @a ) {
	$choice = splice(@a, rand @a, 1);
	print $choice;
}
