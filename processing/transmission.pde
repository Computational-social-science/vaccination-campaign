Table table, point;
color c1=color(255,128,128);
color c2=color(128,192,128);
color c3=color(128,128,255);
color c4=color(223,128,223);
color c5=color(128,233,233);

int lim=800;
int LimLineVal1 =50;
int LimLineVal2 =10;
int LimCicleVal =10;

void setup(){
  //loading
  table = loadTable("./reult-day-0419.csv", "header");
  point = loadTable("./point.csv", "header");
  
  //basic setting
  size(5000,5000);
  background(255);
  pushMatrix();
  scale(5);
  
  //load point csv
  int PCount = point.getRowCount();
  int AllCount = table.getRowCount();
  
  //vector prepare
  PVector []xy = new PVector[PCount];
  
  //load vector
  for(int i =0; i< PCount; i++){
    float xi = point.getInt(i,"x");
    float yi = point.getInt(i,"y");
    int idi = point.getInt(i,"id");
       
    xy[i] = new PVector(xi,yi,idi);
    }

  //draw prepare
  strokeWeight(1);
  colorMode(RGB,255);
  noFill();
  println(PCount);
  //println(xy[0]);
  //println(xy[719]);

  for(int i=0; i<PCount; i++){
    float Va = table.getFloat(i,"weight");
    
    if(Va > 40){
      noStroke();
      fill(220);
      int P2 = int(random(20));
      while(P2<20){
      circle(xy[i].x+random(-40,40),xy[i].y+random(-40,40),5);
      P2++;
        }  
    }
  }

  for(int i=0; i<AllCount; i++){
  
  String S = String.valueOf(table.getInt(i,"source"));
  String T = String.valueOf(table.getInt(i,"target"));
    
  float SW = table.getFloat(int(S)-1,"weight");
  float TW = table.getFloat(int(T)-1,"weight");
  
  if(int(S)<lim && int(T)<lim){
    int fl =255;
    int fl2 =50;
        
      //println(S+":"+SW,T+":"+TW);
        if(table.getInt(i,"target-topic")==0){
          stroke(c1,fl2);
        }
        if(table.getInt(i,"target-topic")==1){
          stroke(c2,fl2);
        }
        if(table.getInt(i,"target-topic")==2){
          stroke(c3,fl2);
        }
        if(table.getInt(i,"target-topic")==3){
          stroke(c4,fl2);
        }
        if(table.getInt(i,"target-topic")==4){
          stroke(c5,fl2);
        }
        int SNo = (table.findRow(S,"id")).getInt("id")-1;
        int TNo = (table.findRow(T,"id")).getInt("id")-1;
        
        
      if((SW> LimLineVal2 & TW> LimLineVal2)){
          if(table.getInt(SNo,"sec") == table.getInt(TNo,"sec")){
            line(xy[SNo].x,xy[SNo].y,xy[TNo].x,xy[TNo].y);
          }
      }
      
        if(table.getInt(i,"target-topic")==0){
          stroke(c1,fl);
        }
        if(table.getInt(i,"target-topic")==1){
          stroke(c2,fl);
        }
        if(table.getInt(i,"target-topic")==2){
          stroke(c3,fl);
        }
        if(table.getInt(i,"target-topic")==3){
          stroke(c4,fl);
        }
        if(table.getInt(i,"target-topic")==4){
          stroke(c5,fl);
        }
        
      if((SW> LimLineVal1 & TW> LimLineVal1)){
          if(table.getInt(SNo,"sec") == 2 && table.getInt(TNo,"sec") == 1){
            DrawMyLineY(xy[SNo].x,xy[SNo].y,xy[TNo].x,xy[TNo].y);
          }
          else if(table.getInt(SNo,"sec") == 1 && table.getInt(TNo,"sec") == 2){
            DrawMyLineY(xy[SNo].x,xy[SNo].y,xy[TNo].x,xy[TNo].y);
          }
          else if(table.getInt(SNo,"sec") == 3 && table.getInt(TNo,"sec") == 1){
            DrawMyLineY(xy[SNo].x,xy[SNo].y,xy[TNo].x,xy[TNo].y);
          }
          else if(table.getInt(SNo,"sec") == 1 && table.getInt(TNo,"sec") == 3){
            DrawMyLineY(xy[SNo].x,xy[SNo].y,xy[TNo].x,xy[TNo].y);
          }
         println(table.getInt(SNo,"sec")+":"+table.getInt(TNo,"sec"));
      }
    }
  }
    
    for(int i=0; i<PCount; i++){
      float Va = table.getFloat(i,"weight");
      //print(Va);
      if(Va>LimCicleVal || i<4){
        //float pro = (Va)/100.0;
        float fl = 255;
        float Wei = (0.6*Va+40)/100;
        //float Wei =1;
        //println(Va+":"+pro);
        if(table.getInt(i,"topic") ==0){
          //strokeWeight(3);
          fill(c1,fl);
        }
        if(table.getInt(i,"topic") ==1){
          fill(c2,fl);
        }
        if(table.getInt(i,"topic") ==2){
          fill(c3,fl);
        }
        if(table.getInt(i,"topic") ==3){
          //strokeWeight(3);
          fill(c4,fl);
        }
        if(table.getInt(i,"topic") ==4){
          //strokeWeight(3);
          fill(c5,fl);
        }
        stroke(0);
        strokeWeight(1);
        circle(xy[i].x,xy[i].y,15*Wei);
      }
    }  
  
  popMatrix();
  save("1.jpg");
}
