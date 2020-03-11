// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <sstream>
// Include GLEW
#include <GL/glew.h>
// Include GLFW
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
using namespace glm;
// Include AntTweakBar
#include <AntTweakBar.h>

#include <common/shader.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include <iostream>

typedef struct Vertex {
	float XYZW[4];
	float RGBA[4];
	void SetCoords(float *coords) {
		XYZW[0] = coords[0];
		XYZW[1] = coords[1];
		XYZW[2] = coords[2];
		XYZW[3] = coords[3];
	}
	void SetColor(float *color) {
		RGBA[0] = color[0];
		RGBA[1] = color[1];
		RGBA[2] = color[2];
		RGBA[3] = color[3];
	}
};

// ATTN: USE POINT STRUCTS FOR EASIER COMPUTATIONS
typedef struct point {
	float x, y, z;
	point(const float x = 0, const float y = 0, const float z = 0) : x(x), y(y), z(z){};
	point(float *coords) : x(coords[0]), y(coords[1]), z(coords[2]){};
	point operator -(const point& a)const {
		return point(x - a.x, y - a.y, z - a.z);
	}
	point operator +(const point& a)const {
		return point(x + a.x, y + a.y, z + a.z);
	}
	point operator *(const float& a)const {
		return point(x*a, y*a, z*a);
	}
	point operator /(const float& a)const {
		return point(x / a, y / a, z / a);
	}
	float* toArray() {
		float array[] = { x, y, z, 1.0f };
		return array;
	}
};

// function prototypes
int initWindow(void);
void initOpenGL(void);
void createVAOs(Vertex[], unsigned short[], size_t, size_t, int);
void createObjects(void);
void pickVertex(void);
void moveVertex(void);
void drawScene(void);
void cleanup(void);
static void mouseCallback(GLFWwindow*, int, int, int);
static void keyCallback(GLFWwindow*, int, int, int, int);

// GLOBAL VARIABLES
GLFWwindow* window;
const GLuint window_width = 1024, window_height = 768;

glm::mat4 gProjectionMatrix;
glm::mat4 gViewMatrix;

GLuint gPickedIndex;
std::string gMessage;
std::string shiftMessage;


GLuint programID;
GLuint pickingProgramID;

// ATTN: INCREASE THIS NUMBER AS YOU CREATE NEW OBJECTS
const GLuint NumObjects = 5;	// number of different "objects" to be drawn
GLuint VertexArrayId[NumObjects] = { 0 };
GLuint VertexBufferId[NumObjects] = { 0 };
GLuint IndexBufferId[NumObjects] = { 0 };
size_t NumVert[NumObjects] = { 0 };

GLuint MatrixID;
GLuint ViewMatrixID;
GLuint ModelMatrixID;
GLuint PickingMatrixID;
GLuint pickingColorArrayID;
GLuint pickingColorID;
GLuint LightID;

Vertex Vertices[] =
        {
                { { 1.0f, 0.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 1
                { { 0.7f, 1.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 2
                { { -0.7f, 1.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 3
                { { -1.0f, 0.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 4
                { { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 5
                { { 1.0f, -0.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 6
                { { 0.7f, -1.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 7
                { { -0.7f, -1.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 8
                { { -1.0f, -0.7f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 9
                { { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 0
        };

unsigned short Indices[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

const size_t IndexCount = sizeof(Indices) / sizeof(unsigned short);

// set this procedurally for greater number of points
float pickingColor[IndexCount] = { 0 / 255.0f, 1 / 255.0f, 2 / 255.0f, 3 / 255.0f,
                                   4 / 255.0f, 5 / 255.0f, 6 / 255.0f, 7 / 255.0f,
                                   8 / 255.0f, 9 / 255.0f};

////////// Variables //////////

///// Picking Code /////
vec3 mousePos;

// For speed computation
double lastTime = glfwGetTime();
int nbFrames = 0;

// Store original color
float oldR = 0.0;
float oldG = 0.0;
float oldB = 0.0;
bool holdingVertex = false;

///// Tasks /////

int subdivisionLevel = 0;
// [num clicks][memory]
Vertex subdivisionVerts[6][IndexCount*32];
Vertex nVertices[10];
// Memory to allot for the new indices
// Weird note: changing this to IndexCount*32 makes showBezier and showCatmull start as true???
unsigned short subdivisionIndices[512];
int prev_num_points = 10;

// Show
bool showBezier = false;
bool showCatmull = false;
bool showDoubleView = false;
bool pickZaxis = false;

// Colors for new points in each of the tasks
float setBlue[] = {0.0f, 1.0f, 1.0f, 1.0f};
float setYellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
float setRed[] = {1.0f, 0.0f, 0.0f, 1.0f};
float setGreen[] = {0.0f, 1.0f, 0.0f, 1.0f};
float setBack[] = {0.0f, 0.0f, 0.4f, 1.0f};

// Bezier indices/vertices
Vertex bezierVerts[40];

// Casteljau points
int points_per_seg = 15;

// Catmull indicies/vertices
// 4 points per catmull * 10 = 40
int numCatmulls = 40;
Vertex catmullVerts[40];
unsigned short catmullIndices[40];
// 15 points per segment * 10 = 150
Vertex casteljau[150];
unsigned short casteljauIndices[150];

float tempYpos;


// ATTN: ADD YOU PER-OBJECT GLOBAL ARRAY DEFINITIONS HERE

void createObjects(void){

    ////////// Task 1: Subdivision //////////


    // Whether or not we show anything
    if(subdivisionLevel != 0) {
        // k = subdivision level
        for(int k = 1; k <= subdivisionLevel; k++) {

            point p_k_2i_plus_1, p_k_2i;

            // Prep the vertices arrays by adding the control points as starters
            for(int j = 0; j < IndexCount; j++) {
                subdivisionVerts[0][j] = Vertices[j];
                subdivisionVerts[0][j].SetColor(setBlue);
            }
            // Connect start with end
            subdivisionVerts[0][IndexCount] = subdivisionVerts[0][0];

            // What was the previous range, i.e. N*2^(k-1) - 1
            prev_num_points = IndexCount * pow(2, k - 1) - 1;

            // i = index
            for(int i = 0; i < (IndexCount * pow(2, subdivisionLevel)) / 2; i++) {

                point *p_k_minus_1_i_minus_1;
                point *p_k_minus_1_i_plus_1;

                auto *p_k_minus_1_i = new point(subdivisionVerts[k-1][i].XYZW);

                if(i == 0) {
                    // If i = 0, i-1 is where we left off last subdivision
                    p_k_minus_1_i_minus_1 = new point(subdivisionVerts[k-1][prev_num_points].XYZW);
                } else {
                    p_k_minus_1_i_minus_1 = new point(subdivisionVerts[k-1][i-1].XYZW);
                }

                if(i > (IndexCount * pow(2, k) - 1)) {
                    p_k_minus_1_i_plus_1 = new point(subdivisionVerts[k-1][1].XYZW);
                } else {
                    p_k_minus_1_i_plus_1 = new point(subdivisionVerts[k-1][i+1].XYZW);
                }

                // Formulas to create a refined set of control points
                p_k_2i_plus_1 = (*p_k_minus_1_i_minus_1 + *p_k_minus_1_i * 6 + *p_k_minus_1_i_plus_1) / 8;
                p_k_2i = (*p_k_minus_1_i_minus_1 * 4 + *p_k_minus_1_i * 4) / 8;

                // Actually put them in
                subdivisionVerts[k][2*i] = {p_k_2i.x, p_k_2i.y, p_k_2i.z, 1.0f};
                subdivisionVerts[k][2*i + 1] = {p_k_2i_plus_1.x, p_k_2i_plus_1.y, p_k_2i_plus_1.z, 1.0f};

                // Set the vertex colors
                subdivisionVerts[k][2*i].SetColor(setBlue);
                subdivisionVerts[k][2*i + 1].SetColor(setBlue);
            }
            subdivisionVerts[k][int(IndexCount * pow(2, subdivisionLevel))] = subdivisionVerts[k][0];
            subdivisionVerts[k][int(IndexCount * pow(2, subdivisionLevel))].SetColor(setBlue);
        }
    }

    ////////// Task 2: Bezier Curves //////////

    // We use our control points, Vertices[]
    if(showBezier) {
        for(int i = 0; i < IndexCount; i++) {
            point *p_i_minus_1, *p_i, *p_i_plus_1, *p_i_plus_2;
            p_i = new point(Vertices[i].XYZW);

            if(i == 0) {
                p_i_minus_1 = new point(Vertices[IndexCount-1].XYZW);
                p_i_plus_1 = new point(Vertices[i + 1].XYZW);
                p_i_plus_2 = new point(Vertices[i + 2].XYZW);
            } else if(i == (IndexCount - 2)){
                p_i_minus_1 = new point(Vertices[i-1].XYZW);
                p_i_plus_1 = new point(Vertices[i+1].XYZW);
                p_i_plus_2 = new point(Vertices[0].XYZW);
            } else if(i == (IndexCount - 1)){
                p_i_minus_1 = new point(Vertices[i-1].XYZW);
                p_i_plus_1 = new point(Vertices[0].XYZW);
                p_i_plus_2 = new point(Vertices[1].XYZW);
            } else {
                p_i_minus_1 = new point(Vertices[i - 1].XYZW);
                p_i_plus_1 = new point(Vertices[i + 1].XYZW);
                p_i_plus_2 = new point(Vertices[i + 2].XYZW);
            }

            // Create a point 1/3 away from p_i, 2/3 away from p_i_plus_1 (equ. 1)
            point c_i_1 = (*p_i * 2 + *p_i_plus_1) / 3;
            bezierVerts[4*i + 1] = {c_i_1.x, c_i_1.y, c_i_1.z, 1.0f};
            bezierVerts[4*i + 1].SetColor(setYellow);

            // Create a point 2/3 away from p_i, 1/3 away from p_i_plus_1 (equ. 2)
            point c_i_2 = (*p_i + *p_i_plus_1 * 2) / 3;
            bezierVerts[4*i + 2] = {c_i_2.x, c_i_2.y, c_i_2.z, 1.0f};
            bezierVerts[4*i + 2].SetColor(setYellow);

            // Need to put C0 and C3 after, since they need C1 and C2 in their calculations

            // C0
            point c_i_0 = (*p_i_minus_1 + *p_i*2) / 3;
            // Midpoint
            c_i_0.x = (c_i_0.x + c_i_1.x) / 2;
            c_i_0.y = (c_i_0.y + c_i_1.y) / 2;
            bezierVerts[4*i] = {c_i_0.x, c_i_0.y, c_i_0.z, 1.0f};
            bezierVerts[4*i].SetColor(setYellow);

            // Create a point 2/3 away from p_i, 1/3 away from p_i_plus_1 (equ. 2)
            point c_i_3 = (*p_i_plus_1 * 2 + *p_i_plus_2) / 3;
            // Midpoint
            c_i_3.x = (c_i_3.x + c_i_2.x) / 2;
            c_i_3.y = (c_i_3.y + c_i_2.y) / 2;
            bezierVerts[4*i + 3] = {c_i_3.x, c_i_3.y, c_i_3.z, 1.0f};
            bezierVerts[4*i + 3].SetColor(setYellow);
        }
    }

    ////////// Task 3: Catmull-rom //////////

    if(showCatmull) {

        // Do catmull
        for (int i = 0; i < IndexCount; i++) {
            point *p_i_minus_1, *p_i, *p_i_plus_1, *p_i_plus_2;
            p_i = new point(Vertices[i].XYZW);

            if(i == 0) {
                p_i_minus_1 = new point(Vertices[IndexCount-1].XYZW);
                p_i_plus_1 = new point(Vertices[i + 1].XYZW);
                p_i_plus_2 = new point(Vertices[i + 2].XYZW);
            } else if(i == (IndexCount - 2)){
                p_i_minus_1 = new point(Vertices[i-1].XYZW);
                p_i_plus_1 = new point(Vertices[i+1].XYZW);
                p_i_plus_2 = new point(Vertices[0].XYZW);
            } else if(i == (IndexCount - 1)){
                p_i_minus_1 = new point(Vertices[i-1].XYZW);
                p_i_plus_1 = new point(Vertices[0].XYZW);
                p_i_plus_2 = new point(Vertices[1].XYZW);
            } else {
                p_i_minus_1 = new point(Vertices[i - 1].XYZW);
                p_i_plus_1 = new point(Vertices[i + 1].XYZW);
                p_i_plus_2 = new point(Vertices[i + 2].XYZW);
            }

            point c_i_0 = *p_i;
            catmullVerts[4*i] = {c_i_0.x, c_i_0.y, c_i_0.z, 1.0f};
            catmullVerts[4*i].SetColor(setRed);

            point c_i_1 = *p_i + (*p_i_plus_1-*p_i_minus_1)/5;
            catmullVerts[4*i + 1] = {c_i_1.x, c_i_1.y, c_i_1.z, 1.0f};
            catmullVerts[4*i + 1].SetColor(setRed);

            point c_i_2 = *p_i_plus_1 - (*p_i_plus_2-*p_i)/5;
            catmullVerts[4*i + 2] = {c_i_2.x, c_i_2.y, c_i_2.z, 1.0f};
            catmullVerts[4*i + 2].SetColor(setRed);

            point c_i_3 = *p_i_plus_1;
            catmullVerts[4*i + 3] = {c_i_3.x, c_i_3.y, c_i_3.z, 1.0f};
            catmullVerts[4*i + 3].SetColor(setRed);
        }

        Vertex temp[numCatmulls];

        // Do casteljau
        for (int i = 0; i < IndexCount; i++) {

            int A = 4 * i;
            int B = 4 * i +1;

            // U at professor peter's suggestion
            for (int u = 0; u < points_per_seg; u++) {

                // So we don't have to mess with catmullVerts directly
                // Fill temp array
                for (int m = 0; m < numCatmulls; m++) {
                    temp[m] = catmullVerts[m];
                }

                // Our "t" for casteljau's
                float t = float(u)/15.0f;

                // Casteljau algorithm
                for (int k = 1; k < 4; k++) {
                    for (int l = 0; l < (4 - k); l++) {
                        temp[A+l]= {
                                (1.0f - t) * (temp[A+l].XYZW[0]) + t*(temp[B+l].XYZW[0]),
                                (1.0f - t) * (temp[A+l].XYZW[1]) + t*(temp[B+l].XYZW[1]),
                                0.0f,
                                1.0f};
                        temp[A+l].SetColor(setGreen);
                    }
                }
                casteljau[(points_per_seg * i) + u] = {
                        temp[A].XYZW[0],
                        temp[A].XYZW[1],
                        temp[A].XYZW[2],
                        temp[A].XYZW[3]
                };
                casteljau[(points_per_seg * i) + u].SetColor(setGreen);
            }
        }
    }
}

void drawScene(void)
{
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	// Re-clear the screen for real rendering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(programID);
	{
		glm::mat4 ModelMatrix = glm::mat4(1.0); // TranslationMatrix * RotationMatrix;
        glm::mat4 ModelMatrix2 = glm::mat4(1.0);  // Second ModelMatrix for Translation

        // If double view mode is enabled, create the layout
        if (showDoubleView) {

            // Scale model1 (which will also scale model2, since model2 will be created using model1)
            ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.65f));
            // Move model1 up
            ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.5f, 0.0f));

            // Create model2 below by setting it equal to model1 moved down
            ModelMatrix2 = glm::translate(ModelMatrix,glm::vec3(0.0f, -4.5f, 0.0f));
            // Rotate model2 around Y Axis by -pi/2, to get side view (negative so that positive is to the right)
            ModelMatrix2 = glm::rotate(ModelMatrix2,float(-3.14159265358979323846 / 2),glm::vec3(0.0f, 1.0f, 0.0f));
        }

        glm::mat4 MVP = gProjectionMatrix * gViewMatrix * ModelMatrix;
        glm::mat4 MVP2 = gProjectionMatrix * gViewMatrix * ModelMatrix2;

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
        glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
        glm::vec3 lightPos = glm::vec3(4, 4, 4);
        glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

		glEnable(GL_PROGRAM_POINT_SIZE);

		// Draw the 10 control points
		glBindVertexArray(VertexArrayId[0]);	// draw Vertices
        glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);				// update buffer data
        glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
        //glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);

        // ATTN: OTHER BINDING AND DRAWING COMMANDS GO HERE, one set per object:

        // Draw each of the inner circle points
        if (subdivisionLevel != 0) {
            glBindVertexArray(VertexArrayId[1]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[1]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivisionVerts[subdivisionLevel]),
                            subdivisionVerts[subdivisionLevel]);
            glDrawElements(GL_POINTS, NumVert[1], GL_UNSIGNED_SHORT, (void *) 0);
            glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void *) 0);
        }

        if(showBezier) {
            glBindVertexArray(VertexArrayId[2]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[2]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(bezierVerts), bezierVerts);
            glDrawElements(GL_POINTS, 40, GL_UNSIGNED_SHORT, (void *) 0);
            glDrawElements(GL_LINE_LOOP, 40, GL_UNSIGNED_SHORT, (void *) 0);
        }

        if(showCatmull){
            glBindVertexArray(VertexArrayId[3]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[3]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(catmullVerts), catmullVerts);
            glDrawElements(GL_LINE_LOOP, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
            glDrawElements(GL_POINTS, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);

            glBindVertexArray(VertexArrayId[4]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[4]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(casteljau), casteljau);
            glDrawElements(GL_LINE_LOOP, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
        }

        // If double view is enabled, draw the second set of points
        if (showDoubleView) {

            // Draw ModelMatrix2
            glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP2[0][0]);
            glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix2[0][0]);
            glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
            glm::vec3 lightPos = glm::vec3(4, 4, 4);
            glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

            glEnable(GL_PROGRAM_POINT_SIZE);

            // Draw the 10 control points
            glBindVertexArray(VertexArrayId[0]);	// draw Vertices
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);				// update buffer data
            glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
            //glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);

            // ATTN: OTHER BINDING AND DRAWING COMMANDS GO HERE, one set per object:

            // Draw each of the inner circle points
            if (subdivisionLevel != 0) {
                glBindVertexArray(VertexArrayId[1]);
                glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[1]);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivisionVerts[subdivisionLevel]),
                                subdivisionVerts[subdivisionLevel]);
                glDrawElements(GL_POINTS, NumVert[1], GL_UNSIGNED_SHORT, (void *) 0);
                glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void *) 0);
            }

            if(showBezier) {
                glBindVertexArray(VertexArrayId[2]);
                glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[2]);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(bezierVerts), bezierVerts);
                glDrawElements(GL_POINTS, 40, GL_UNSIGNED_SHORT, (void *) 0);
                glDrawElements(GL_LINE_LOOP, 40, GL_UNSIGNED_SHORT, (void *) 0);
            }

            if(showCatmull){
                glBindVertexArray(VertexArrayId[3]);
                glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[3]);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(catmullVerts), catmullVerts);
                glDrawElements(GL_LINE_LOOP, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
                glDrawElements(GL_POINTS, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);

                glBindVertexArray(VertexArrayId[4]);
                glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[4]);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(casteljau), casteljau);
                glDrawElements(GL_LINE_LOOP, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
            }
        }

        glBindVertexArray(0);
	}
	glUseProgram(0);
	// Draw GUI
	TwDraw();

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void pickVertex(void)
{
	// Clear the screen in white
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(pickingProgramID);
	{
		glm::mat4 ModelMatrix = glm::mat4(1.0); // TranslationMatrix * RotationMatrix;
		// If we're changing the view, we'll have to adjust accordingly so we pick the right place
		if (showDoubleView){
            ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.65f));
            ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.5f, 0.0f));
        }
		glm::mat4 MVP = gProjectionMatrix * gViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, in the "MVP" uniform
		glUniformMatrix4fv(PickingMatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniform1fv(pickingColorArrayID, NumVert[0], pickingColor);	// here we pass in the picking marker array

		// Draw the ponts
		glEnable(GL_PROGRAM_POINT_SIZE);
		glBindVertexArray(VertexArrayId[0]);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);	// update buffer data
		glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
		glBindVertexArray(0);
	}
	glUseProgram(0);
	// Wait until all the pending drawing commands are really done.
	// Ultra-mega-over slow ! 
	// There are usually a long time between glDrawElements() and
	// all the fragments completely rasterized.
	glFlush();
	glFinish();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Read the pixel at the center of the screen.
	// You can also use glfwGetMousePos().
	// Ultra-mega-over slow too, even for 1 pixel, 
	// because the framebuffer is on the GPU.
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	unsigned char data[4];
	glReadPixels(xpos, window_height - ypos, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data); // OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top

    // If we aren't holding a vertex, check what we just clicked on
    if (!holdingVertex) {
        // Convert the color back to an integer ID
        gPickedIndex = int(data[0]);
    }

    if (gPickedIndex == 255){ // Full white, must be the background !
        gMessage = "background";
    }
    else {
        std::ostringstream oss;
        oss << "point " << gPickedIndex;
        gMessage = oss.str();

        if (gPickedIndex < IndexCount) {
            // If we're not currently holding a vertex, store the picked vertex's color info
            if (!holdingVertex) {
                oldR = Vertices[gPickedIndex].RGBA[0];
                oldG = Vertices[gPickedIndex].RGBA[1];
                oldB = Vertices[gPickedIndex].RGBA[2];
                // Store the y position so we can move freely about y-axis
                tempYpos = Vertices[gPickedIndex].XYZW[1];
                holdingVertex = true;
            }

            // Apply color change (old color but darker, to give a highlight effect)
            Vertices[gPickedIndex].RGBA[0] = oldR/3;
            Vertices[gPickedIndex].RGBA[1] = oldG/3;
            Vertices[gPickedIndex].RGBA[2] = oldB/3;
        }
    }
}

// fill this function in!
void moveVertex(void)
{
	glm::mat4 ModelMatrix = glm::mat4(1.0);
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    // If we're changing the view, we'll have to adjust accordingly so we pick the right place
    if (showDoubleView){
        ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.65f));
        ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.5f, 0.0f));
    }
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glm::vec4 vp = glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]);

    // If we are holding a vertex, set the vertex's position to the mouse position (drag vertex)
    if (holdingVertex) {

        float zpos = 0.0;
        // Unproject converts window coordinates to 3D scene position
        mousePos = glm::unProject(glm::vec3(window_width - xpos, window_height - ypos, zpos), ModelMatrix, gProjectionMatrix,
                                  vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
        if (!pickZaxis) {
            Vertices[gPickedIndex].XYZW[0] = mousePos.x;
            Vertices[gPickedIndex].XYZW[1] = mousePos.y;
        }
        else{
            Vertices[gPickedIndex].XYZW[2] = mousePos.y-tempYpos;
        }
    }

    if (gPickedIndex == 255){ // Full white, must be the background !
        gMessage = "background";
    }
    else {
        std::ostringstream oss;
        oss << "point " << gPickedIndex;
        gMessage = oss.str();

        for (int i = 0; i <= IndexCount; i++) {
            if (i == 9) {
                nVertices[i].SetCoords(Vertices[0].XYZW);
            }
            else {
                nVertices[i].SetCoords(Vertices[i].XYZW);
            }
        }
    }

}

int initWindow(void)
{
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // FOR MAC

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(window_width, window_height, "Michael DelSole (01380402)", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Initialize the GUI
	TwInit(TW_OPENGL_CORE, NULL);
	TwWindowSize(window_width, window_height);
	TwBar * GUI = TwNewBar("Picking");
	TwSetParam(GUI, NULL, "refresh", TW_PARAM_CSTRING, 1, "0.1");
	TwAddVarRW(GUI, "Last picked object", TW_TYPE_STDSTRING, &gMessage, NULL);
    TwAddVarRW(GUI, "Shift is:", TW_TYPE_STDSTRING, &shiftMessage, NULL);


    // Set up inputs
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_FALSE);
	glfwSetCursorPos(window, window_width / 2, window_height / 2);
	glfwSetMouseButtonCallback(window, mouseCallback);
    glfwSetKeyCallback(window, keyCallback);

	return 0;
}

void initOpenGL(void)
{
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

    // Bezier
	for(int i = 0; i < 512; i++) {
        subdivisionIndices[i] = i;
    }

    // Catmull-Rom
    for (int i = 0; i < 4*IndexCount; i++) {
        catmullIndices[i] = i;
    }

    // Casteljau
    for (int i = 0; i < points_per_seg*IndexCount; i++) {
        casteljauIndices[i] = i;
    }

	// Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	//glm::mat4 ProjectionMatrix = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
	// Or, for an ortho camera :
	gProjectionMatrix = glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.0f, 100.0f); // In world coordinates

	// Camera matrix
	gViewMatrix = glm::lookAt(
		glm::vec3(0, 0, -5), // Camera is at (4,3,3), in World Space
		glm::vec3(0, 0, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
		);

	// Create and compile our GLSL program from the shaders
    programID = LoadShaders("hw1bShade.vertexshader", "hw1bShade.fragmentshader");
    pickingProgramID = LoadShaders("hw1bPick.vertexshader", "hw1bPick.fragmentshader");

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");
	ViewMatrixID = glGetUniformLocation(programID, "V");
	ModelMatrixID = glGetUniformLocation(programID, "M");
	PickingMatrixID = glGetUniformLocation(pickingProgramID, "MVP");
	// Get a handle for our "pickingColorID" uniform
	pickingColorArrayID = glGetUniformLocation(pickingProgramID, "PickingColorArray");
	pickingColorID = glGetUniformLocation(pickingProgramID, "PickingColor");
	// Get a handle for our "LightPosition" uniform
	LightID = glGetUniformLocation(programID, "LightPosition_worldspace");

	createVAOs(Vertices, Indices, sizeof(Vertices), sizeof(Indices), 0);
	createObjects();

	// ATTN: create VAOs for each of the newly created objects here:

    createVAOs(subdivisionVerts[subdivisionLevel], subdivisionIndices, sizeof(subdivisionVerts[subdivisionLevel]), sizeof(subdivisionIndices), 1);
    createVAOs(bezierVerts, subdivisionIndices, sizeof(bezierVerts), sizeof(subdivisionIndices), 2);
    createVAOs(catmullVerts, catmullIndices, sizeof(catmullVerts), sizeof(catmullIndices), 3);
    createVAOs(casteljau, casteljauIndices, sizeof(casteljau), sizeof(casteljauIndices), 4);

}

void createVAOs(Vertex Vertices[], unsigned short Indices[], size_t BufferSize, size_t IdxBufferSize, int ObjectId) {

	NumVert[ObjectId] = IdxBufferSize / (sizeof(GLubyte));

	GLenum ErrorCheckValue = glGetError();
	size_t VertexSize = sizeof(Vertices[0]);
	size_t RgbOffset = sizeof(Vertices[0].XYZW);

	// Create Vertex Array Object
	glGenVertexArrays(1, &VertexArrayId[ObjectId]);
	glBindVertexArray(VertexArrayId[ObjectId]);

	// Create Buffer for vertex data
	glGenBuffers(1, &VertexBufferId[ObjectId]);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[ObjectId]);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, Vertices, GL_STATIC_DRAW);

	// Create Buffer for indices
	glGenBuffers(1, &IndexBufferId[ObjectId]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ObjectId]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, IdxBufferSize, Indices, GL_STATIC_DRAW);

	// Assign vertex attributes
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);

	glEnableVertexAttribArray(0);	// position
	glEnableVertexAttribArray(1);	// color

	// Disable our Vertex Buffer Object 
	glBindVertexArray(0);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
			);
	}
}

void cleanup(void)
{
	// Cleanup VBO and shader
	for (int i = 0; i < NumObjects; i++) {
		glDeleteBuffers(1, &VertexBufferId[i]);
		glDeleteBuffers(1, &IndexBufferId[i]);
		glDeleteVertexArrays(1, &VertexArrayId[i]);
	}
	glDeleteProgram(programID);
	glDeleteProgram(pickingProgramID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
{
	// If the mouse has just been pressed
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		pickVertex();
	}

    // If the mouse has just been released
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        // If we're holding a vertex
        if (holdingVertex && gPickedIndex < IndexCount) {
            // Drop it and restore the vertex's old colors
            Vertices[gPickedIndex].RGBA[0] = oldR;
            Vertices[gPickedIndex].RGBA[1] = oldG;
            Vertices[gPickedIndex].RGBA[2] = oldB;
            Vertices[gPickedIndex].RGBA[3] = 1.0f;
            holdingVertex = false;
        }
    }
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if(key == GLFW_KEY_1 && action == GLFW_PRESS) {
        if(subdivisionLevel < 5) {
            subdivisionLevel += 1;
        } else {
            subdivisionLevel = 0;
        }
    }

    if(key == GLFW_KEY_2 && action == GLFW_PRESS) {
        showBezier = !showBezier;
    }

    if(key == GLFW_KEY_3 && action == GLFW_PRESS) {
        showCatmull = !showCatmull;
    }

    if(key == GLFW_KEY_4 && action == GLFW_PRESS) {
        showDoubleView = !showDoubleView;
    }

    // Shift key Z-Axis Movement Toggle
    if ((key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) && action== GLFW_PRESS) {

        pickZaxis = true;
        std::cout<<pickZaxis<<"\n";
        shiftMessage = "Held down";

    }
    else if ((key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) && action== GLFW_RELEASE){
        pickZaxis = false;
        shiftMessage = "NOT held down";

    }
}

int main(void)
{
	// initialize window
	int errorCode = initWindow();
	if (errorCode != 0)
		return errorCode;

	// initialize OpenGL pipeline
	initOpenGL();

	// For speed computation
	double lastTime = glfwGetTime();
	int nbFrames = 0;
	do {
		// Measure speed
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0){ // If last prinf() was more than 1sec ago
			// printf and reset
			printf("%f ms/frame\n", 1000.0 / double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}

		// DRAGGING: move current (picked) vertex with cursor
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT))
			moveVertex();

		// DRAWING SCENE
		createObjects();	// re-evaluate curves in case vertices have been moved
		drawScene();


	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);

	cleanup();

	return 0;
}
