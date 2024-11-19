function adamW(parr,gtin,mtin,vtin,tspin) {
	
	//calculate vec adjust
	mtin = b1*mtin + (1-b1)*gtin;//get first vec change
	vtin = b2*vtin + (1-b2)*pow(gtin,2);//get second vec change
	let mtv = mtin/(1-pow(b1,tspin));//first vec bias correct
	let vtv = vtin/(1-pow(b2,tspin));//second vec bias correct

	//send back values
	return [mtin,vtin,alpha*mtv/(sqrt(vtv)+epsilon)];

}//network consts
