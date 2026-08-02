#version 450 core

layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(std430,binding=0) readonly buffer Input { float data[]; } input_buffer;
layout(std430,binding=1) writeonly buffer Output { float data[]; } output_buffer;
layout(push_constant) uniform Parameters { uint iw; uint ih; uint ow; uint oh; } p;
float read_value(uint x,uint y){return input_buffer.data[y*p.iw+x];}
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;if(x>=p.ow||y>=p.oh)return;float sx=clamp((float(x)+0.5)*float(p.iw)/float(p.ow)-0.5,0.0,float(p.iw-1));float sy=clamp((float(y)+0.5)*float(p.ih)/float(p.oh)-0.5,0.0,float(p.ih-1));uint x0=uint(floor(sx)),y0=uint(floor(sy)),x1=min(x0+1,p.iw-1),y1=min(y0+1,p.ih-1);float top=mix(read_value(x0,y0),read_value(x1,y0),sx-float(x0));float bottom=mix(read_value(x0,y1),read_value(x1,y1),sx-float(x0));output_buffer.data[y*p.ow+x]=mix(top,bottom,sy-float(y0));}
