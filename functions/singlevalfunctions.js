function GELU(num) {
	
	return scale*num/(1+pow(e,-1.702*num));
	
}

function sigmoid(num) {
	
	return (1/(1+(pow(e,-1*scale*num))))
	
}

function RELU(num) {
	
	return max(0,scale*num);
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
}

function getfuncderiv(input) {
		
	if (type == "sigmoid") {
		return (scale*pow(e,-1*scale*input))/pow(1+pow(e,-1*scale*input),2);
	}
	else if (type == "RELU") {
		return 1;
	}
	else if (type == "GELU") {
		let ndx = pow(e,1.702*input);
		return (ndx*scale*(1+ndx+1.702*input))/pow(1+ndx,2)
	}

}

function untoken(Q) {
	
	for (g = 0; g < tokens.length; g++) {
		if (tokens[g] == Q) {
			return g;
		}
	}
	if (Q == "") {
		return 8;
	}//newline catch
	return -1;
	
}

function positioners(x) {
	
	let sx = learningset/1.57079633;
	return sin(x/sx);
	
}
