function activate(ARR,type,scale) {
	
	let shapeARR = shape(ARR);
	let actfunc = function(parr,inputs) {
		if (type == "sigmoid") {
			return sigmoid1(dimen(false,inputs[0],parr),1);
		}
		else if (type == "RELU") {
			return RELU1(dimen(false,inputs[0],parr),scale);
		}
		else if (type == "GELU") {
			return GELU1(dimen(false,inputs[0],parr),scale);
		}
	}
	return maketensor(shapeARR.length,shapeARR,actfunc,[ARR]);
	
}

function softmax(ARR,smtemperature) {
	
	let shapeARR = shape(ARR);
	shapeARR = shapeARR.slice(0,shapeARR.length-1);
	let smin = function(parr,inputs) {
		let touse = dimen(false,inputs[0],parr);
		let exsum = 0;
		let arrtoreturn = [];
		for (let g = 0; g < touse.length; g++) {
			exsum += pow(2.718281828459045,touse[g]/smtemperature);
		}
		for (let g = 0; g < touse.length; g++) {
			arrtoreturn[g] = pow(2.718281828459045,touse[g]/smtemperature)/exsum;
			if (isNaN(arrtoreturn[g])) {
				arrtoreturn[g] = 1;
			}
		}
		return arrtoreturn;
	}
	
	if (shapeARR.length == 0) {
		return smin([0],[[ARR]]);
	}
	return maketensor(shapeARR.length,shapeARR,smin,[ARR]);
	
}

function Bsort(ARR,sourceARR,softmaxb,highlow,byprop,prop) {
	
	if (softmaxb == true) {
		ARR = softmax(ARR);
	}
	
	let sorted = [];
	if (byprop != true) {
		sorted = transpose([sourceARR,ARR]);
	}
	else {
		sorted = ARR;
	}
	
	let arrp = 1;
	if (byprop == true) {
		arrp = prop;
	}
	if (highlow == true) {
		sorted.sort(function(p, q) {
			return q[arrp] - p[arrp];
		});
	}
	else {
		sorted.sort(function(p, q) {
			return p[arrp] - q[arrp];
		});
	}

	return sorted;
	
}//[0] is decoded value, [1] is strength of that value

function normalize(ARR,scalar) {
	
	let shapeARR = shape(ARR);
	shapeARR = shapeARR.slice(0,shapeARR.length-1);
	let normin = function(parr,inputs) {
		let touse = dimen(false,inputs[0],parr);
		let arrneg = opxd("mult",CA(touse),tensor(-1,[touse.length]));
		let nv = max(max(touse),max(arrneg))/scalar;//find maximum value
		return opxd("div",CA(touse),tensor(nv,[touse.length]));
	}
	
	if (shapeARR.length == 0) {
		return normin([0],[[ARR]]);
	}
	return maketensor(shapeARR.length,shapeARR,normin,[ARR]);
	
}

function contains(ARR,val) {

	let shapeARR = shape(ARR);
	let conin = function(parr,inputs) {
		if (inputs[1] == false) {
			inputs[1] = (dimen(false,inputs[0],parr)==val);
		}
		return 0;
	}
	let inarr = [ARR,false];
	maketensor(shapeARR.length,shapeARR,conin,inarr);
	return inarr[1];
	
}
