function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {

	if (shapeARR.length > dim) {
		shapeARR = shapeARR.slice(0,dim);
	}
	else if (shapeARR.length < dim) {
		for (let a = shapeARR.length; a < dim; a++) {
			shapeARR[a] = 0;
		}
	}
	let inparr = [];	
	function recurmt(parr) {
		let ra = []
		if (parr.length > 0) {
			for (let a = 0; a < parr[0]; a++) {
				inparr[dim-parr.length] = a;
				ra[a] = recurmt(parr.slice(1,parr.length));
			}
			return ra;
		}
		else {
			return getfill(inparr);
		}
	}
	
	function getfill(parr) {
		if (ifrand === true) {
			if (ifroundrand == true) {
				return rr(randl,randh+1);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending === true) {
			return parr[parr.length-1];
		}
		else if (typeof fill === 'function') {
			return fill(parr,ifrand);//input arr gets assigned to ifrand
		}
		else if (fill !== undefined && fill[0] !== undefined) {
			return CA(fill);
		}//array
		else {
			return fill;
		}
	}
	
	return recurmt(shapeARR);
	
}

function shapenet(shapeARR,specific,dim,hidlay,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let totalshape;
	if (specific == false) {
		totalshape = [shapeARR[0]];
		for (let gsn = 1; gsn < hidlay+1; gsn++) {
			totalshape[gsn] = shapeARR[1];
		}
		totalshape[hidlay+1] = shapeARR[2];
	}
	else {
		totalshape = shapeARR;
	}
	let rasn = [];
	for (let gsn = 0; gsn < totalshape.length-1; gsn++) {
		rasn[gsn] = maketensor(dim,[totalshape[gsn+int(dim==1)],totalshape[gsn+1]],fill,ifrand,randl,randh,ifroundrand,ascending);
	}
	return rasn;
	
}

function dimen(assign,arr,parr,val) {
	if (parr.length > 0) {
		if (assign == true) {
			arr[parr[0]] = dimen(true,arr[parr[0]],parr.slice(1,parr.length),val);
			return arr;
		}
		else {
			return dimen(false,arr[parr[0]],parr.slice(1,parr.length))
		}
	}
	else {
		if (assign == true) {
			return val;
		}
		else {
			return arr;
		}
	}
}//different dimensional arrays- assign: bool- t:assign or f:return

function opxd(oper,ARR1,ARR2) {
	
	let shapeARR = shape(ARR1);
	if (shapeARR.join(",") != shape(ARR2).join(",")) {
		print("misaligned opxd shapes");
		exit()
	}
	
	function opfunc(val1,val2) {
		if (oper == "add") {
			return val1+val2;
		}
		if (oper == "sub") {
			return val1-val2;
		}
		if (oper == "mult") {
			return val1*val2;
		}
		if (oper == "div") {
			return val1/val2;
		}
	}
	
	function opxdloop(arr1,arr2,parr) {
		if (parr.length > 0) {
			for (let a = 0; a < parr[0]; a++) {
				arr1[a] = opxdloop(arr1[a],arr2[a],parr.slice(1,parr.length));
			}
			return arr1;
		}
		else {
			return opfunc(arr1,arr2);
		}
	}
	
	return opxdloop(ARR1,ARR2,shapeARR);
	
}//adds two same size arrays

function shape(ARR1) {

  let shapeARR = [];
	testv = ARR1;
	for (let ga = 0; testv[0] !== undefined; ga++) {
		shapeARR[ga] = testv.length;
    testv = testv[0];
	}
  return shapeARR;

}

function tensor(fill,shapearr) {

  return maketensor(shapearr.length,shapearr,fill);

}

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

function CA(ARR) {

	return JSON.parse(JSON.stringify(ARR));;
	
}//prevents awful javascript auto-pointers (copy array)
