void DrawMyLineX(float x1, float y1, float x2, float y2){
  noFill();
  if(random(1)<0.5){
  bezier(x2, y2, x2, 4*(y1+y2)/7*random(0.9,1.1), x1, 4*(y1+y2)/7*random(0.9,1.1), x1, y1);
  }
  else{
  bezier(x1, y1, x1, 3*(y1+y2)/7*random(0.9,1.1), x2, 3*(y1+y2)/7*random(0.9,1.1), x2, y2);
  }
}
