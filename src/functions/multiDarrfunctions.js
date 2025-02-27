function transpose(ARR) {
	
	let returntps = tensor(0,[ARR[0].length,ARR.length]);
	for (let a = 0; a < ARR.length; a++) {
		for (let b = 0; b < ARR[a].length; b++) {
			returntps[b][a] = CA([ARR[a][b]])[0];
		}
	}
	return returntps;
	
}//switch columns with rows, preserve values

function concatenate(ARR,dims) {
	
	if (dims === undefined) {
		dims = 1;
	}
	let ccatin = function(arr) {
		let ra = [];
		for (let g = 0; g < arr.length; g++) {
			for (let g1 = 0; g1 < arr[g].length; g1++) {
				ra[ra.length] = arr[g][g1];
			}
		}
		return ra;
	}
	for (let a = 0; a < dims; a++) {
		ARR = ccatin(CA(ARR));
	}
	return ARR;
	
}//combines rows of 2d array into 1d array

function CA(ARR,obj) {
	
	return ARR.slice(0);
	
}//prevents awful javascript auto-pointers (copy array)
