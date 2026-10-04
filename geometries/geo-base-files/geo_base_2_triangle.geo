

//inner geometry
Point(1) = {x1, y1, 0, size};
Point(2) = {x2, y2, 0, size};
Point(3) = {x3, y3, 0, size};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Curve Loop(1) = {1, 2, 3};
//Plane Surface(1) = {1};

//outer geometry
Point(5) = {0, 0, 0, size};
Point(6) = {260 * unit, 0, 0, size};
Point(7) = {260 * unit, 120 * unit, 0, size};
Point(8) = {0, 120 * unit, 0, size};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2, 1};

//Transfinite Surface {2};
Recombine Surface {2};

//Now make 3D by extrusion.
out[] = 
Extrude { 0,0,0.01 }
{
	Surface{2};
	Layers{1};
	Recombine;
};

Physical Surface("frontAndBack") = {2, out[0]};
Physical Surface("outlet") = {out[3]};
Physical Surface("fixedWalls") = {out[4], out[2], out[6], out[7], out[8]};
Physical Surface("inlet") = {out[5]};
Physical Volume(100) = {out[1]};


