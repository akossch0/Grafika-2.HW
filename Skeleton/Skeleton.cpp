//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Schneider Ákos
// Neptun : XYUXUA
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
const float epsilon = 0.0001f;
const float lightHeight = 0.98f;
const float height = lightHeight;
const float roomHeight = 2.1f;
const float radius = sqrtf(powf(roomHeight, 2) * (1 - powf(height, 2)));
const float A = powf(radius, 2.0f) * M_PI;
const int sampleNr = 1;
const int numberOfTestPoints = 5;
enum MaterialType{ROUGH, REFLECTIVE};
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	MaterialType type;
	vec3 F0;
	
	Material(MaterialType t) {
		type = t;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x/denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa): Material(REFLECTIVE){
		vec3 one(1.0f, 1.0f, 1.0f);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Quadrics {
protected:
	Material* material;
public:
	mat4 Q; //symmetric matrix

	float f(vec4 r) { //r.w = 1
		return dot(r * Q, r);
	}

	virtual Hit intersect(const Ray& ray) = 0;
};


struct Ellipsoid : public Quadrics {
	vec3 center, size;

	Ellipsoid(vec3 s, vec3 c, Material* mat): size(s), center(c){
		material = mat;

		float x = (-2.0f * center.x) / (size.x * size.x);
		float ex = (center.x * center.x) / (size.x * size.x);
		float y = (-2.0f * center.y) / (size.y * size.y);
		float ey = (center.y * center.y) / (size.y * size.y);
		float z = (-2.0f * center.z) / (size.z * size.z);
		float ez = (center.z * center.z) / (size.z * size.z);

		Q = mat4(vec4(1/(size.x*size.x), 0.0f, 0.0f, 0.0f),
				 vec4(0.0f, 1/(size.y*size.y), 0.0f, 0.0f),
				 vec4(0.0f, 0.0f, 1/(size.z*size.z), 0.0f),
				 vec4(x, y, z, -1.0f + ex + ey + ez));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit hit2;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1.0f);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		
		float d = ((b * b) - (4.f * a * c));
		if (d < 0.f || a == 0.f || b == 0.f || c == 0.f)
			return hit;
		d = sqrt(d);
		float t1 = (-b + d) / (2.f * a);
		float t2 = (-b - d) / (2.f * a);

		if (t1 <= epsilon && t2 <= epsilon) return hit; // ha mindketto interszekcio a ray kiindulopont mogott van
		bool back = (t1 <= epsilon || t2 <= epsilon); // ha csak az egyik t>0 akkor az ellipszoid belsejeben vagyunk 
														
		float t = 0.f;
		if (t1 <= epsilon)
			t = t2;
		else
			if (t2 <= epsilon)
				t = t1;
			else
				t = (t1 < t2) ? t1 : t2;

		if (t < epsilon) return hit; // Too close to intersection
		hit.t = t;
		hit.position = ray.start + hit.t * ray.dir;
		if (hit.position.z > lightHeight) return hit2; //lyukvagas
		vec4 vec((hit.position - center).x, (hit.position - center).y, (hit.position - center).z, 1.0f);
		hit.normal = hit.position - center;
		hit.normal.x = 2.f * hit.normal.x / (size.x * size.x);
		hit.normal.y = 2.f * hit.normal.y / (size.y * size.y);
		hit.normal.z = 2.f * hit.normal.z / (size.z * size.z);
		float f = (back) ? -1.f : 1.f;
		hit.normal = hit.normal * f;
		hit.normal = normalize(hit.normal);
		
		hit.material = material;
		return hit;
	}
};

struct Cylinder : public Quadrics {
	vec2 size;
	vec3 center;
	float z_max, z_min;
	
	Cylinder(vec2 s, vec3 c, Material* mat, float _zmx, float _zmn) : size(s), center(c), z_max(_zmx), z_min(_zmn) {
		material = mat;

		float x = (-2.0f * center.x) / (size.x * size.x);
		float x2 = (center.x * center.x) / (size.x * size.x);
		float y = (-2.0f * center.y) / (size.y * size.y);
		float y2 = (center.y * center.y) / (size.y * size.y);
		
		Q = mat4(vec4(1 / (size.x * size.x), 0.0f, 0.0f, 0.0f),
				 vec4(0.0f, 1 / (size.y * size.y), 0.0f, 0.0f),
				 vec4(0.0f, 0.0f, 0.0f, 0.0f),
				 vec4(x, y, 0.0f, -1.0f + x2 + y2));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit hit2;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1.0f);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0 && ray.dir.z) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= epsilon && t2 <= epsilon) return hit; 
		bool back = (t1 <= epsilon || t2 <= epsilon); 
														
		float t = 0.f;
		float t_inner = 0.f;
		if (t1 <= epsilon)
			t = t2;
		else
			if (t2 <= epsilon)
				t = t1;
			else {
				t = (t1 < t2) ? t1 : t2;
				t_inner = (t1 < t2) ? t2 : t1;
			}
		if (t < epsilon) return hit; // Too close to intersection
		hit.t = t;
		hit.position = ray.start + hit.t * ray.dir;
		if (hit.position.z > z_max || hit.position.z < z_min) {
			
			if (hit.position.z > z_max) {
				hit.t = t_inner;
				hit.position = ray.start + hit.t * ray.dir;
				if (hit.position.z > z_max) {
					return hit2;
				}
				else {
					hit.normal = vec3(0, 0, 1);
					hit.material = material;
					return hit;
				}
			}
			return hit2;
		}
		hit.normal = hit.position -	center;
		hit.normal.x = 2.f * hit.normal.x / (size.x * size.x);
		hit.normal.y = 2.f * hit.normal.y / (size.y * size.y);
		hit.normal.z = 2.f * hit.normal.z;

		hit.normal = hit.normal * ((back) ? -1.f : 1.f);
		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

struct Hiperboloid : public Quadrics {
	vec3 center;
	vec3 size;
	float z_max, z_min;

	Hiperboloid(vec3 c, vec3 s, Material* mat, float _z_max, float _z_min) : center(c), size(s), z_max(_z_max), z_min(_z_min){
		material = mat;

		float x = (-2.0f * center.x) / (size.x * size.x);
		float ex = (center.x * center.x) / (size.x * size.x);
		float y = (-2.0f * center.y) / (size.y * size.y);
		float ey = (center.y * center.y) / (size.y * size.y);
		float z = (2.0f * center.z) / (size.z * size.z);
		float ez = -1.0f * (center.z * center.z) / (size.z * size.z);

		Q = mat4(vec4(1 / (size.x * size.x), 0.0f, 0.0f, 0.0f),
			vec4(0.0f, 1 / (size.y * size.y), 0.0f, 0.0f),
			vec4(0.0f, 0.0f, -1.0f/(size.z * size.z), 0.0f),
			vec4(x, y, z, -1.0f + ex + ey + ez));
	}

	Hit intersect(const Ray& ray) {
		Hit hit, hit2;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1.0f);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= epsilon && t2 <= epsilon) return hit; 
		bool back = (t1 <= epsilon || t2 <= epsilon); 
														
		float t = 0.f;
		float t_inner = 0.f;
		if (t1 <= epsilon)
			t = t2;
		else
			if (t2 <= epsilon)
				t = t1;
			else {
				t = (t1 < t2) ? t1 : t2;
				t_inner = (t1 < t2) ? t2 : t1;
			}
				

		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + hit.t * ray.dir;
		if (hit.position.z > z_max || hit.position.z < z_min) {
			if (hit.position.z < z_min) {
				hit.t = t_inner;
				hit.position = ray.start + hit.t * ray.dir;
				if (hit.position.z < z_min) {
					return hit2;
				}
				else {
					hit.normal = hit.position - center;
					hit.normal.x = 2.f * hit.normal.x / (size.x * size.x);
					hit.normal.y = 2.f * hit.normal.y / (size.y * size.y);
					hit.normal.z = 2.f * hit.normal.z;

					hit.normal = hit.normal * ((back) ? -1.f : 1.f);
					hit.normal = normalize(hit.normal);
					hit.material = material;
					return hit;
				}
			}
			if (hit.position.z > z_max) {
				hit.t = t_inner;
				hit.position = ray.start + hit.t * ray.dir;
				if (hit.position.z > z_max) {
					return hit2;
				}
				else {
					hit.normal = vec3(0, 0, 1);
					hit.material = material;
					return hit;
				}
			}
			return hit2;
		}
		hit.normal = hit.position - center;
		hit.normal.x = 2.f * hit.normal.x / (size.x * size.x);
		hit.normal.y = 2.f * hit.normal.y / (size.y * size.y);
		hit.normal.z = 2.f * hit.normal.z;

		hit.normal = hit.normal * ((back) ? -1.f : 1.f);
		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		fov = fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }
float RandomFloat(float a, float b) {
	float random = rnd();
	float diff = b - a;
	float r = random * diff;
	return a + r;
}
class Scene {
	std::vector<Quadrics*> objects;
	std::vector<Light*> lights;

	Camera camera;
	vec3 La;
	vec3 sunDirection, sunLe;
	vec3 skyLe;
public:
	void build() {
		//eye(0.0f, 1.85f, -0.4f)
		//szoba kivulrol: eye(0.0,2.05,2.0), lookat(0,0,1)
		vec3 eye = vec3(0.0f, 1.85f, -0.4f), vup = vec3(0, 0, 1.0f), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180; //field of view
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.17f);
		//sun
		sunDirection = vec3(0.0f, -1.0f, 0.6f);
		sunLe = vec3(1.0f, 1.0f, 1.0f);
		
		//sky
		skyLe = vec3(1.5f, 1.5f, 1.5f);

		//szoba szine
		vec3 kd(0.25f, 0.25f, 0.25f), ks(0.1f,0.1f,0.1f);

		//Arany: n / k : 0.17 / 3.1, 0.35 / 2.7, 1.5 / 1.9
		vec3 gold_n(0.17f, 0.35f, 2.7f);
		vec3 gold_kappa(3.1f, 2.7f, 1.9f);

		//Ezust: n/k: 0.14/4.1, 0.16/2.3, 0.13/3.1
		vec3 silver_n(0.14f, 0.16f, 0.13f);
		vec3 silver_kappa(4.0f, 2.3f, 3.1f);

		Material* material = new RoughMaterial(kd, ks, 50);
		Material* gold = new ReflectiveMaterial(gold_n, gold_kappa);
		Material* silver = new ReflectiveMaterial(silver_n, silver_kappa);
		Material* blue = new RoughMaterial(vec3(0.1f, 0.2f, 0.3f), vec3(0.2f,0.2f,0.2f), 50);
		Material* red = new RoughMaterial(vec3(0.3f, 0.1f, 0.1f), vec3(0.3f,0.3f,0.3f), 50);
		Material* pink = new RoughMaterial(vec3(0.43f, 0.22f, 0.29f), vec3(0.3f, 0.3f, 0.3f), 50);
		Material* roughgold = new RoughMaterial(vec3(0.5f, 0.4215f, 0.0f), vec3(0.3f, 0.3f, 0.3f), 50);
		Material* green = new RoughMaterial(vec3(0.1f, 0.2f, 0.1f), ks, 50);

		//the objects
		//henger
		objects.push_back(new Cylinder(vec2(0.15f,0.15f), vec3(-0.4f, 1.0f,0.0f), blue, -0.6f, -1.0f));

		//szoba
		objects.push_back(new Ellipsoid(vec3(2.1f, 2.1f, 1.0f), vec3(0.0f,0.0f,0.0f), material));

		//aranytukor
		objects.push_back(new Ellipsoid(vec3(1.2f, 0.8f, 1.6f), vec3(-0.2f, -0.7f, -1.4f), gold));

		//rozsaszin ellipszoid
		objects.push_back(new Ellipsoid(vec3(0.15f, 0.15f, 1.1f), vec3(-1.7f, -0.3f, -1.0f), pink));

		//kis zold hiperboloid
		objects.push_back(new Hiperboloid(vec3(0.25f, 1.0f, -1.0f), vec3(0.05f, 0.05f, 0.1f), green, -0.7f, -1.0f));

		//nagy piros hiperboloid
		objects.push_back(new Hiperboloid(vec3(0.9f, 0.5f, -0.5f), vec3(0.1f, 0.1f, 0.27f), red, 0.35f, -1.0f));

		//fenysugarcso
		objects.push_back(new Hiperboloid(vec3(0.0f, 0.0f, lightHeight), vec3(radius, radius, 0.5f), silver, 1.6f, lightHeight));

		//arany henger
		objects.push_back(new Cylinder(vec2(0.2f, 0.2f), vec3(1.2f, -1.0f, 0.0f), roughgold, 0.2f, -1.0f));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color(0, 0, 0);

				//hogy az elek szebbek legyenek
				for(int i = 0; i < sampleNr; i++)
					color = color + trace(camera.getRay(X, Y));

				color = color / sampleNr;

				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Quadrics* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Quadrics* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	
	std::vector<vec3> randomPoints(int numberOfTestPoints) {
		std::vector<vec3> result;

		int n = 0;
		for (int i = 0; n < numberOfTestPoints; i++) {
			vec3 randomPointOnSquare = vec3(RandomFloat(-1.0f*radius, radius), RandomFloat(-1.0f*radius, radius), 0.95f);
			vec3 vector = randomPointOnSquare - vec3(0.0f, 0.0f, 0.95f);
			if (length(vector) < radius) {
				result.push_back(randomPointOnSquare);
				n++;
			}
		}
		return result;
	}
	
	vec3 trace(Ray ray, int refdepth = 0, int roughdepth = 0) {

		if (refdepth > 4) return La;

		if (roughdepth > 1) return vec3(0, 0, 0);

		Hit hit = firstIntersect(ray);

		if (hit.t < 0) return skyLe + sunLe * powf(dot(ray.dir, sunDirection), 10);

		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			std::vector<vec3> points = randomPoints(numberOfTestPoints);
			for (auto point : points) {
				Ray shadowRay(hit.position + hit.normal * epsilon, point);
				float cosTheta = dot(hit.normal, normalize(point - hit.position));

				if (cosTheta > 0) {	// shadow computation
					vec3 tmp(0, 0, 0);
					vec3 nor = vec3(0, 0, -1);
					float cosT = dot(hit.position - point, normalize(nor));
					float r = length(point - hit.position);
					float dOmega = (A / points.size()) * (cosT / r * r);
					tmp = trace(Ray(hit.position + hit.normal * epsilon, point - hit.position), refdepth, roughdepth + 1);
					outRadiance = outRadiance + tmp * hit.material->kd * cosTheta * dOmega;
					vec3 halfway = normalize(-ray.dir + point);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0);
						outRadiance = outRadiance + tmp * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1.0f, 1.0f, 1.0f);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), refdepth + 1, roughdepth) * F;

		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao, textureId = 0;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
