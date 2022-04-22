//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
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

const float MASS = 1;
const float CHARGE = 5;

const char * const vertexSource = R"(
	#version 330
	precision highp float;

    uniform mat4 MVP;
	layout(location = 0) in vec4 vp;

	void main() {
        float w = (vp.x, vp.y, sqrt((vp.x * vp.x) + (vp.y * vp.y) + 1));
		gl_Position = vec4(vp.x/(w+1), vp.y/(w+1), 0, 1) * MVP;
	}
)";
const char * const fragmentSource = R"(
	#version 330
	precision highp float;

	uniform vec3 color;
	out vec4 outColor;

	void main() {
		outColor = vec4(color, 1);
	}
)";

unsigned int vao;
unsigned int vbo;
GPUProgram gpuProgram;
const int resolution = 100;

/*https://stackoverflow.com/questions/686353/random-float-number-generation*/
float randomNum(float low, float high) {return  low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (high - low)));}
struct Camera {
    vec2 wCenter;
    Camera(): wCenter(0, 0){}
    mat4 SlideMatrix() { return TranslateMatrix(wCenter);}
    void Slide(vec2 t) { wCenter = wCenter + t; }
};
Camera camera;
struct Atom{
    vec4 points[resolution];
    std::vector<float> drag;
    float charge, radius, mass;
    vec4 center;
    vec3 color;

    Atom(float rangeX, float rangeY){
        float x,y;
        x = randomNum(rangeX - 0.9f, rangeX + 0.9f);
        y = randomNum(rangeY - 0.9f, rangeY + 0.9f);

        center = vec4(x,y,0,1);
        mass = (rand() % 100 + 1) * MASS;
        radius = fmax(mass * 0.001, 0.08);
    }
    void draw(){
        for(int i = 0; i < resolution; i++){
            float fi = i * 2 * M_PI / resolution;
            points[i] = center + radius * (vec4(cosf(fi), sinf(fi), 0, 1));
        }
        mat4 MVPTransform = camera.SlideMatrix();
        gpuProgram.setUniform(color, "color");
        gpuProgram.setUniform(MVPTransform, "MVP");
        glBufferData(GL_ARRAY_BUFFER, resolution * sizeof(vec4), &points[0], GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLE_FAN, 0, resolution);
    }
    void setCharge(float charge){
        this->charge = charge;
        this->color = vec3(fmax(charge * 0.003, 0.0), 0, -fmin(charge * 0.003 , 0));
    }
};
struct Line{
    vec4 points[resolution];
    vec4 q, p;
    Line(vec4 start, vec4 end) {
        this->p = start;
        this->q = end;
    }
    void draw(){
        vec2 vector = vec2(q.x-p.x, q.y-p.y);
        for (int i = 0; i <= resolution; i++)
            points[i] = vec4(p.x + (i * vector.x / resolution), p.y + (i * vector.y / resolution), 0, 1);
        mat4 MVPTransform = camera.SlideMatrix();
        gpuProgram.setUniform(vec3(1, 1, 1), "color");
        gpuProgram.setUniform(MVPTransform, "MVP");
        glBufferData(GL_ARRAY_BUFFER, resolution * sizeof(vec4), &points[0], GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, resolution);
    }
};
struct Molecule{
    std::vector<Atom*> atoms;
    std::vector<Line*> lines;
    vec4 sp = vec4(0.0f,0.0f, 0.0f, 0.0f);
    vec2 velocity = vec2(0.0f,0.0f);
    vec2 shift = vec2(0.0f,0.0f);
    vec2 vel = vec2(0.0f,0.0f);
    float omega = 0;
    float theta = 0;
    float rotation = 0;
    Molecule(){
        int numOfCircles = rand() % 6 + 3;
        float rangeX = randomNum(-1.0f, 1.0f);
        float rangeY = randomNum(-1.0f, 1.0f);

        for(int i = 0; i < numOfCircles; i++) {
            if(i == 0) {
                atoms.push_back(new Atom(rangeX, rangeY));
                continue;
            }
            Atom *a = new Atom(rangeX, rangeY);
            lines.push_back(new Line(a->center, atoms[rand() % atoms.size()]->center));
            atoms.push_back(a);
        }
        setTheta();
        setSp();
        setCharges();
    }
    void draw(){
        for(auto l : lines){l->draw();}
        for(auto a : atoms){a->draw();}
    }
    void setSp(){
        vec4 top;
        float bottom;
        for(auto a : atoms){
            top += a->center * a->mass;
            bottom += a->mass;
        }
        sp = top/bottom;
    }
    void setTheta(){
        for(auto a : atoms) {
            vec2 spToCenter(a->center.x - sp.x, a->center.y - sp.y);
            float len = powf(length(spToCenter), 2.0f);
            theta += len + a->mass;
        }
    }
    void setCharges(){
        float charge = 0;
        float sum = 0;
        for(int i = 0; i < atoms.size() - 1; i++){
            charge = (rand() % 100 - 50) * CHARGE;
            sum += charge;
            atoms[i]->setCharge(charge);
            if(i == atoms.size() - 2){
                atoms[i+1]->setCharge(-sum);
                break;
            }
        }
    }
    float getMass(){
        float sum = 0;
        for(auto a : atoms) {
            sum += a->mass;
        }
        return sum;
    }
    void origoMove(int multiplier){
        for (auto a : atoms){
            a->center = a->center - (this->sp * multiplier);
        }
        for (auto l : lines){
            l->p = l->p - (this->sp * multiplier);
            l->q = l->q - (this->sp * multiplier);
        }
    }
    void move(vec3 shift){
        mat4 mtxShift = TranslateMatrix(shift);
        for(auto a : atoms){
            vec4 center = vec4(a->center.x, a->center.y, 0, 1) * mtxShift;
            a->center.x = center.x;
            a->center.y = center.y;
        }
        for(auto l : lines){
            vec4 center = vec4(l->p.x , l->p.y, 0, 1) * mtxShift;
            l->p.x = center.x;
            l->p.y = center.y;

            center = vec4(l->q.x , l->q.y, 0, 1) * mtxShift;
            l->q.x = center.x;
            l->q.y = center.y;
        }
    }
    void rotate(float angle){
        vec3 sp = vec3(this->sp.x, this->sp.y, 1);
        mat4 mtxRotate = RotationMatrix(angle,sp);
        for(auto a : atoms){
            vec4 center = vec4(a->center.x, a->center.y, 0, 1) * mtxRotate;
            a->center.x = center.x;
            a->center.y = center.y;
        }
        for(auto l : lines){
            vec4 center = vec4(l->p.x , l->p.y, 0, 1) * mtxRotate;
            l->p.x = center.x;
            l->p.y = center.y;

            center = vec4(l->q.x , l->q.y, 0, 1) * mtxRotate;
            l->q.x = center.x;
            l->q.y = center.y;
        }
    }
};
std::vector<Molecule*> molecules;

const float adjustRotation = 0.005f;
const float adjustMovement = 0.0001f;
void animate(float dt){
    for(int i = 0; i < molecules.size(); i++){
        vec2 direction(0.0f, 0.0f);
        vec2 Mpush(0.0f, 0.0f);
        vec2 cF(0.0f, 0.0f);
        float Mrotate = 0;

        molecules[i]->setTheta();
        molecules[i]->setSp();
        molecules[i]->origoMove(1);

        for(int j = 0; j < molecules[i]->atoms.size(); j++){
            Atom *curr = molecules[i]->atoms[j];
            int other = (i == 0) ? 1 : 0;
            for(int x = 0; x < molecules[other]->atoms.size(); x++){
                vec2 curr2(curr->center.x, curr->center.y);
                vec2 other2(molecules[other]->atoms[x]->center.x, molecules[other]->atoms[x]->center.y);
                direction = vec2(other2 - curr2);
                float r = length(direction);
                cF = cF + ((curr->charge * molecules[other]->atoms[x]->charge) / 2.0f * float(M_PI) * 0.01f * r) * normalize(direction);
            }
            vec2 sp2(molecules[i]->sp.x, molecules[i]->sp.y);
            vec2 center2(molecules[i]->atoms[j]->center.x, molecules[i]->atoms[j]->center.y);
            vec2 spToCenter(center2 - sp2);
            vec2 dF = molecules[i]->velocity * spToCenter * 0.47f;
            cF = cF - dF;
            Mpush = Mpush + cF;
            Mrotate = Mrotate + cross(spToCenter, cF).z;
        }
        molecules[i]->velocity = molecules[i]->velocity + Mpush / molecules[i]->getMass() * dt;
        molecules[i]->shift = molecules[i]->shift + molecules[i]->velocity * dt * adjustMovement;

        molecules[i]->omega = molecules[i]->omega + Mrotate / molecules[i]->theta * dt;
        molecules[i]->rotation = molecules[i]->rotation + molecules[i]->omega * dt * adjustRotation;

        molecules[i]->move(molecules[i]->shift);
        molecules[i]->rotate(molecules[i]->rotation);
        molecules[i]->origoMove(-1);
    }
}

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    gpuProgram.create(vertexSource, fragmentSource, "outColor");
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for(auto m : molecules){m->draw();}
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case 's':
            camera.Slide(vec2(-0.1, 0));
            break;
        case 'd':
            camera.Slide(vec2(+0.1, 0));
            break;
        case 'x':
            camera.Slide(vec2( 0,-0.1));
            break;
        case 'e':
            camera.Slide(vec2( 0, +0.1));
            break;
        case ' ':
            molecules.clear();
            molecules.push_back(new Molecule());
            molecules.push_back(new Molecule());
            break;
    }
    glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
    static float tend = 0;
    const float dt = 0.01;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME)/1000.0f;
    for(float t = tstart; t < tend; t += dt){
        float Dt = std::min(dt, tend - t);
        animate(Dt);
    }
    glutPostRedisplay();
}
