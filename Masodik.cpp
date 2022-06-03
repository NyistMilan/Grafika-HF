//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
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
// Nev    : Nyist Milan Konor
// Neptun : VU9J1J
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
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

struct Material{
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};
struct Hit{
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};
struct Ray{
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
struct Intersectable{
    Material * material;
    virtual Hit intersect(const Ray& ray) = 0;
};
struct Quadrics : public Intersectable{
    mat4 Q;
    vec3 gradf(vec4 r) {
        r.w = 1;
        vec4 g = r * Q * 2;
        return vec3(g.x, g.y, g.z);
    }
    mat4 transpose(mat4 matrix) {
        int rows = 4;
        int columns = 4;
        mat4 transpose;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                transpose[i][j] = matrix[j][i];
        return transpose;
    }
    void rotate(float angle, vec3 v){Q = RotationMatrix(angle, v) * Q * transpose(RotationMatrix(angle, v));}
    void transform(vec3 v){Q = TranslateMatrix(v) * Q * transpose(TranslateMatrix(v));}
    void scale(vec3 v) {Q = ScaleMatrix(v) * Q * transpose(ScaleMatrix(v));}
};

struct Paraboloid : public Quadrics{
    vec3 focus;
    float top;

    Paraboloid(vec3 _focus, float _top, Material* _material){
        Q = mat4(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 0, 1,
                 0, 0, 1, 0);;
        top = _top;
        material = _material;
        focus = _focus;

        this->scale(vec3(2,4,1));
        this->transform(vec3(0,-5.4,0));
        this->rotate(M_PI / -2, vec3(0, 1, 0));
    }
    Hit intersect(const Ray& ray) {
        Hit hit;
        vec4 norm_dir = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        vec4 norm_start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        float a = dot(norm_dir * Q, norm_dir);
        float b = 2.0f * dot(norm_start * Q, norm_dir);
        float c = dot(norm_start * Q, norm_start);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float t1 = (-b + sqrtf(discr)) / (2.0f * a);
        float t2 = (-b - sqrtf(discr)) / (2.0f * a);
        if (t1 <= 0) return hit;
        vec3 hitPos1 = ray.start + ray.dir * t1;
        vec3 hitPos2 = ray.start + ray.dir * t2;

        float dist1 = length(hitPos1 - focus);
        if(dist1 > top)
            t1 = -1;
        float dist2 = length(hitPos2 - focus);
        if(dist2 > top)
            t2 = -1;

        if (t1 < 0.0 && t2 < 0.0){
            hit.t = -1.0;
        } else if (t2 < 0.0){
            hit.t = t1;
            hit.position = hitPos1;
        } else if (t1 < 0.0){
            hit.t = t2;
            hit.position = hitPos2;
        } else{
            if (t1 < t2) {
                hit.t = t1;
                hit.position = hitPos1;
            } else{
                hit.t = t2;
                hit.position = hitPos2;
            }
        }
        hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1)));
        hit.material = material;
        return hit;
    }
};
struct Cylinder : public Intersectable{
    vec3 center;
    float radius, height;

    Cylinder(vec3 _center, float _radius, float _height, Material* _material){
        center = _center;
        radius = _radius;
        height = _height;
        material = _material;
    }
    Hit intersect(const Ray& ray){
        Hit hit;
        vec2 oc = vec2(ray.start.x,ray.start.z)  - vec2(center.x, center.z);
        float a = dot(vec2(ray.dir.x, ray.dir.z), vec2(ray.dir.x, ray.dir.z));
        float b = 2.0 * dot(oc, vec2(ray.dir.x, ray.dir.z));
        float c = dot(oc, oc) - radius * radius;
        float disc = b * b - 4 * a * c;
        if (disc < 0.0){
            return hit;
        }
        float t1 = (-b - sqrtf(disc)) / (2 * a);
        float t2 = (-b + sqrtf(disc)) / (2 * a);
        vec3 hitPos1 = ray.start + ray.dir * t1;
        vec3 hitPos2 = ray.start + ray.dir * t2;
        if (hitPos1.y < center.y || hitPos1.y > height + center.y)
            t1 = -1.0;
        if (hitPos2.y < center.y || hitPos2.y > height + center.y)
            t2 = -1.0;

        if (t1 < 0.0 && t2 < 0.0){
            hit.t = -1.0;
        } else if (t2 < 0.0){
            hit.t = t1;
            hit.position = hitPos1;
        } else if (t1 < 0.0){
            hit.t = t2;
            hit.position = hitPos2;
        } else{
            if (t1 < t2) {
                hit.t = t1;
                hit.position = hitPos1;
            } else{
                hit.t = t2;
                hit.position = hitPos2;
            }
        }
        hit.normal = hit.position - center;
        hit.normal.y = 0.0;
        hit.normal = normalize(hit.normal);
        hit.material = material;
        return hit;
    }
};
struct Sphere : public Intersectable{
    vec3 center;
    float radius;

    Sphere(const vec3& _center, float _radius, Material* _material){
        center = _center;
        radius = _radius;
        material = _material;
    }
    Hit intersect(const Ray& ray){
        Hit hit;
        vec3 oc = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = 2.0 * dot(oc, ray.dir);
        float c = dot(oc, oc) - radius * radius;
        float disc = b * b - 4 * a * c;
        if (disc < 0.0) {
            return hit;
        }
        hit.t = (-b - sqrtf(disc)) / (2 * a);
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(hit.position - center);
        hit.material = material;
        return hit;
    }
};
struct Plane : public Intersectable{
    vec3 point, normal;
    bool cut;

    Plane(const vec3& _point, const vec3& _normal, bool _cut, Material* mat){
        point = _point;
        normal = normalize(_normal);
        cut = _cut;
        material = mat;
    }
    Hit intersect(const Ray& ray) {
        Hit hit;
        double NdotV = dot(normal, ray.dir);
        if (fabs(NdotV) < epsilon) return hit;
        double t = dot(normal, point - ray.start) / NdotV;
        if (t < epsilon) return hit;
        hit.t = t;
        hit.position = ray.start + ray.dir * hit.t;
        if (cut && dot((hit.position - point), (hit.position - point)) > 4){
            hit.t = -1;
            return hit;
        }
        hit.normal = normal;
        if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1); // flip the normal, we are inside the sphere
        hit.material = material;
        return hit;
    }
};

struct Camera{
    vec3 eye, lookat, right, up;
    float fov;

    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        fov = _fov;
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
    void animate(float dt){
        eye = vec3((eye.x - lookat.x) * cosf(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cosf(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }
};
struct Light{
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = _direction;
        Le = _Le;
    }
};
struct Scene{
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;

    void build() {
        vec3 eye = vec3(0, 15, 8), vup = vec3(0, 1, 0), lookat = vec3(0,0, 0);
        float fov = 90 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.05f, 0.05f, 0.05f);
        vec3 lightPos(10, 10, 10);
        vec3 lampPos(-1.0f, 5.4f, 0);
        lights.push_back(new Light(lightPos, vec3(100, 100, 100)));
        lights.push_back(new Light(lampPos, vec3(50,50,50)));

        Material* planeMaterial = new Material(vec3(1.0f, 0.4f, 0.2f), vec3(0.1f, 0.1f, 0.1f), 50);
        Material* lampMaterial = new Material(vec3(0.3f, 0.5f, 0.8f), vec3(0.5f, 0.5f, 0.5f), 50);

        objects.push_back(new Plane(vec3(0, 0, 0), vec3(0, 1, 0), false, planeMaterial));
        objects.push_back(new Cylinder(vec3(0, 0, 0), 2, 0.5, lampMaterial));
        objects.push_back(new Plane(vec3(0, 0.5, 0), vec3(0, 1, 0), true, lampMaterial));
        objects.push_back(new Sphere(vec3(0, 0.5, 0), 0.3, lampMaterial));
        objects.push_back(new Cylinder(vec3(0, 0.2, 0), 0.2, 2.5, lampMaterial));
        objects.push_back(new Sphere(vec3(0, 2.7, 0), 0.3, lampMaterial));
        objects.push_back(new Cylinder(vec3(0, 2.9, 0), 0.2, 2.5, lampMaterial));
        objects.push_back(new Sphere(vec3(0, 5.4, 0), 0.3, lampMaterial));
        objects.push_back(new Paraboloid(vec3(0.0, 5.5f, 0), 2, lampMaterial));
    }
    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }
    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }
    bool shadowIntersect(Ray ray, vec3 light) {	// for directional lights
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && length(light - ray.start) >  length(hit.position - ray.start)){
                return true;
            }
        }
        return false;
    }
    vec3 trace(Ray ray){
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        vec3 outRadiance = hit.material->ka * La;
        for (Light * light : lights) {
            vec3 lightPosition = normalize(light->direction - hit.position + hit.normal * epsilon);
            Ray shadowRay(hit.position + hit.normal * epsilon, lightPosition);
            float cosTheta = dot(hit.normal, lightPosition);
            if (cosTheta > 0 && !shadowIntersect(shadowRay, light->direction)) {	// shadow computation
                outRadiance = outRadiance + (light->Le * hit.material->kd * cosTheta) / length(light->direction - hit.position) / length(light->direction - hit.position);
                vec3 halfway = normalize(-ray.dir + lightPosition);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + (light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess)) / length(light->direction - hit.position) / length(light->direction - hit.position);
            }
        }
        return outRadiance;
    }
    void animate(float dt){camera.animate(dt);}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
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

void onInitialization(){
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
void onDisplay(){
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}
void onKeyboard(unsigned char key, int pX, int pY){}
void onKeyboardUp(unsigned char key, int pX, int pY){}
void onMouse(int button, int state, int pX, int pY){}
void onMouseMotion(int pX, int pY){}
void onIdle(){
    scene.animate(0.1f);
    glutPostRedisplay();
}