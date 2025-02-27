function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let inparr = [];
	function recurmt(mtdim,mtshape) {
		let ra = [];
		for (let g = 0; g < mtshape[0]; g++) {
			inparr[dim-mtdim] = g;
			if (mtdim > 1) {
				ra[g] = recurmt(mtdim-1,mtshape.slice(1,mtshape.length));
			}
			else {
				ra[g] = getfill(inparr);
			}
		}
		return ra;
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
	
	return recurmt(dim,CA(shapeARR));
	
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
	if (parr.length > 1) {
		if (assign == true) {
			arr[parr[0]] = dimen(true,arr[parr[0]],parr.slice(1,parr.length),val);
		}
		else {
			return dimen(false,arr[parr[0]],parr.slice(1,parr.length))
		}
	}
	else {
		if (assign == true) {
			arr[parr[0]] = val;
			return arr;
		}
		else {
			return arr[parr[0]];
		}
	}
}//different dimensional arrays- assign: bool- t:assign or f:return

function opxd(oper,ARR1,ARR2) {
	
	let shapeARR = shape(ARR1);
	if (shapeARR+"" != shape(ARR2)+"") {
		print("misaligned opxd shapes");
		exit()
	}
	
	let opxdin = function opfill(parr,inputs) {
		if (inputs[2] == "add") {
			return dimen(false,inputs[0],parr)+dimen(false,inputs[1],parr);
		}
		if (inputs[2] == "sub") {
			return dimen(false,inputs[0],parr)-dimen(false,inputs[1],parr);
		}
		if (inputs[2] == "mult") {
			return dimen(false,inputs[0],parr)*dimen(false,inputs[1],parr);
		}
		if (inputs[2] == "div") {
			return dimen(false,inputs[0],parr)/dimen(false,inputs[1],parr);
		}
	}
	return maketensor(shapeARR.length,shapeARR,opxdin,[ARR1,ARR2,oper]);
	
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
