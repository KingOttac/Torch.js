function add2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] += ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function sub2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] -= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function mult2d(ARR1,ARR2) {

	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] *= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function div2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] /= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays
