function tril(ARR,v) {
	
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[0].length; g1++) {
			if (g1 > g) {
				ARR[g][g1] = v;
			}
		}
	}
	
	return ARR;
	
}//create triangle of (v) values in 2d array

function transpose(ARR) {
	
	return ARR[0].map((_, colIndex) => ARR.map(row => row[colIndex]));
	
}//switch columns with rows, preserve values

function matrixmult(ARR1,ARR2) {
	
	if (ARR1[0].length != ARR2.length) {
		print("ERROR:\ninfo:\nARR1:",ARR1,"ARR2:",ARR2)
		exit()
	}
	let returnarr = [];
	for (g = 0; g < ARR1.length; g++) {
		returnarr[g] = [];
		for (g1 = 0; g1 < ARR2[0].length; g1++) {
			returnarr[g][g1] = 0;
		}
	}//initialize returnarrs
	
	for (g = 0; g < returnarr.length; g++) {
		for (g1 = 0; g1 < returnarr[g].length; g1++) {
			for (g2 = 0; g2 < ARR1[0].length; g2++) {
				returnarr[g][g1] += ARR1[g][g2]*ARR2[g2][g1];
			}
		}
	}
	
	return returnarr;
	
}//can also be used with ([vec1],transpose([vec2])) for dot product

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

function concatenate(ARR) {
	
	let ra = [];
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[g].length; g1++) {
			ra[ra.length] = ARR[g][g1];
		}
	}
	return ra;
	
}//combines rows of 2d array into 1d array

function CA(ARR,obj) {
	
	return ARR.slice(0);
	
}//prevents awful javascript auto-pointers (copy array)

function dimen(assign,arr,parr,val) {
	let p0 = parr[0];
	let p1 = parr[1];
	let p2 = parr[2];
	let p3 = parr[3];
	let p4 = parr[4];
	let p5 = parr[5];
	let dst = parr.length;
	if (assign == true) {
		if (dst == 1) {
			arr[p0] = val;
		}//1D
		else if (dst == 2) {
			arr[p0][p1] = val;
		}//2D
		else if (dst == 3) {
			arr[p0][p1][p2] = val;
		}//3D
		else if (dst == 4) {
			arr[p0][p1][p2][p3] = val;
		}//4D
		else if (dst == 5) {
			arr[p0][p1][p2][p3][p4] = val;
		}//5D
		else if (dst == 6) {
			arr[p0][p1][p2][p3][p4][p5] = val;
		}//6D
	}
	else {
		if (dst == 1) {
			return arr[p0];
		}//1D
		else if (dst == 2) {
			return arr[p0][p1];
		}//2D
		else if (dst == 3) {
			return arr[p0][p1][p2];
		}//3D
		else if (dst == 4) {
			return arr[p0][p1][p2][p3];
		}//4D
		else if (dst == 5) {
			return arr[p0][p1][p2][p3][p4];
		}//5D
		else if (dst == 6) {
			return arr[p0][p1][p2][p3][p4][p5];
		}//6D
	}
}//different dimensional arrays- assign: bool- t:assign or f:return
