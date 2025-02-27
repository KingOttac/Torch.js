let e = 2.718281828459045

function GELU(num,scl) {
	
	if (scl == undefined) {
		scl = scale;
	}//back compat
	return scl*num/(1+pow(e,-1.702*num));
	
}

function sigmoid(num,scl) {
	
	if (scl == undefined) {
		scl = scale;
	}//back compat
	return (1/(1+(pow(e,-1*scl*num))))
	
}

function RELU(num,scl) {
	
	if (scl == undefined) {
		scl = scale;
	}//back compat
	return max(0,scl*num);
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
}

function getfuncderiv(input,scl) {
	
	if (scl == undefined) {
		scl = scale;
	}//back compat
		
	if (type == "sigmoid") {
		return (scl*pow(e,-1*scl*input))/pow(1+pow(e,-1*scl*input),2);
	}
	else if (type == "RELU") {
		return scl;
	}
	else if (type == "GELU") {
		let ndx = pow(e,1.702*input);
		return (ndx*scl*(1+ndx+1.702*input))/pow(1+ndx,2)
	}

}

function untoken(Q) {
	
	for (let g = 0; g < tokens.length; g++) {
		if (tokens[g] == Q) {
			return g;
		}
	}
	return -1;
	
}

function positioners(x) {
	
	let sx = learningset/1.57079633;
	return sin(x/sx);
	
}
