function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let inparr = [];
	function recurmt(mtdim,mtshape) {
		let ra = [];
		for (let g = 0; g < shapeARR[0]; g++) {
			inparr[dim-mtdim] = g;
			if (mtdim > 1) {
				ra[g] = recurmt(mtdim-1,shapeARR.slice(1,shapeARR.length));
			}
			else {
				ra[g] = getfill(inparr);
			}
		}
		return ra;
	}
	
	function getfill(parr) {
		if (ifrand == true) {
			if (ifroundrand == true) {
				return rr(randl,randh+1);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending == true) {
			return parr[parr.length-1];
		}
		else if (typeof fill === 'function') {
			return fill(parr);
		}
		else if (typeof fill === 'object') {
			return CA(fill);
		}
		else {
			return fill;
		}
	}
	
	return recurmt(dim,CA(shapeARR));
	
}//limit of 6 dimensions, randl = lower bound, randh = upper bound

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
	
	let opxdin = function opfill(parr) {};
	switch (oper) {
		case "add":
			opxdin = function opfill(parr) {
				return dimen(false,ARR1,parr)+dimen(false,ARR2,parr);
			};
		break;
		case "sub":
			opxdin = function opfill(parr) {
				return dimen(false,ARR1,parr)-dimen(false,ARR2,parr);
			};
		break;
		case "mult":
			opxdin = function opfill(parr) {
				return dimen(false,ARR1,parr)*dimen(false,ARR2,parr);
			};
		break;
		case "div":
			opxdin = function opfill(parr) {
				return dimen(false,ARR1,parr)/dimen(false,ARR2,parr);
			};
		break;
	}
	return maketensor(shapeARR.length,shapeARR,opxdin);
	
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
