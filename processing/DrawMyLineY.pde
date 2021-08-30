void DrawMyLineY(float x1, float y1, float x2, float y2){
  noFill();
  if(random(1)<0.5){
  bezier(x1, y1, 3*(x1+x2)/7*random(0.9,1.1), y1, 3*(x1+x2)/7*random(0.9,1.1), y2, x2, y2);
  }
  else{
  bezier(x2, y2, 4*(x1+x2)/7*random(0.9,1.1), y2, 4*(x1+x2)/7*random(0.9,1.1), y1, x1, y1);
  }
}
