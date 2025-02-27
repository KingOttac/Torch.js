function tril(ARR,v) {
	
	for (let g = 0; g < ARR.length; g++) {
		for (let g1 = 0; g1 < ARR[0].length; g1++) {
			if (g1 > g) {
				ARR[g][g1] = v;
			}
		}
	}
	
	return ARR;
	
}//create triangle of (v) values in 2d array

function matrixmult(ARR1,ARR2) {
	
	if (ARR1[0].length != ARR2.length) {
		print("ERROR:\ninfo:\nARR1:",ARR1,"ARR2:",ARR2)
		exit()
	}
	let returnarr = maketensor(2,[ARR1.length,ARR2[0].length],0)
	
	for (let g = 0; g < returnarr.length; g++) {
		for (let g1 = 0; g1 < returnarr[g].length; g1++) {
			for (let g2 = 0; g2 < ARR1[0].length; g2++) {
				returnarr[g][g1] += ARR1[g][g2]*ARR2[g2][g1];
			}
		}
	}
	
	return returnarr;
	
}//can also be used with ([vec1],transpose([vec2])) for dot product

function add2d(ARR1,ARR2) {

	return opxd("add",ARR1,ARR2);
	
}

function sub2d(ARR1,ARR2) {

	return opxd("sub",ARR1,ARR2);
	
}

function mult2d(ARR1,ARR2) {

	return opxd("mult",ARR1,ARR2);
	
}

function div2d(ARR1,ARR2) {

	return opxd("div",ARR1,ARR2);
	
}
