#include "GLUTWrapper.h"
#ifdef __APPLE__
#include <GLUT/GLUT.h>
#else
#include <GL/freeglut.h>
#endif
#include <assert.h>
#include <unistd.h>

GLUTWrapper *GLUTWrapper::wrapper = NULL;

GLUTWrapper::GLUTWrapper():mouseButtons(0), callbacks(NULL) {
	winId = glutCreateWindow("");
	glutDisplayFunc(GLUTWrapper::displayFunc);
	glutReshapeFunc(GLUTWrapper::reshapeFunc);
	glutMouseFunc(GLUTWrapper::mouseFunc);
	glutMotionFunc(GLUTWrapper::mouseMotionFunc);
	glutKeyboardFunc(GLUTWrapper::keyboardFunc);
}

GLUTWrapper *GLUTWrapper::create(int *argc, char *argv[]) {
	assert (wrapper == NULL);
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
	int displayWidth = glutGet(GLUT_SCREEN_WIDTH);
	int displayHeight = glutGet(GLUT_SCREEN_HEIGHT);
	glutInitWindowSize(displayWidth*.6, displayHeight*.6);

	return wrapper = new GLUTWrapper();
}

void GLUTWrapper::setCallbacks(GLUTCallbacks *cb)
{
	callbacks = cb;
}

void GLUTWrapper::setTitle(const std::string &s) {
	glutSetWindowTitle(s.c_str());
}

void GLUTWrapper::displayFunc()
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->display();
	glutSwapBuffers();
}

void GLUTWrapper::reshapeFunc(int w, int h)
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->reshape(w, h);
}

void GLUTWrapper::mouseFunc(int button, int state, int x, int y)
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	if (state == GLUT_DOWN)
		wrapper->mouseButtons |= 1<<button;
	if (state == GLUT_UP)
		wrapper->mouseButtons &= ~(1<<button);
	wrapper->callbacks->mouse(wrapper->mouseButtons, x, y);
}

void GLUTWrapper::mouseMotionFunc(int x, int y)
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->mouse(wrapper->mouseButtons, x, y);
}

void GLUTWrapper::keyboardFunc(unsigned char c, int x, int y)
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->keyPressed(c);
}

void GLUTWrapper::timerFunc(int dummy)
{
	assert (dummy == -1);
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->timer();
}

void GLUTWrapper::startTimer(int time)
{
	glutTimerFunc(time, GLUTWrapper::timerFunc, -1);
}

void GLUTWrapper::postRedraw() {
	glutPostRedisplay();
}

void GLUTWrapper::run() {
	glutMainLoop();
}

void GLUTWrapper::exit(int status) {
	glutDestroyWindow (winId);
	_exit (status);
}


int GLUTWrapper::createSubmenu(GLUTMenuEntry &e) {
	assert (e.size() > 0);

	/* First create all sub-menus */
	std::vector<int> submenuIds;
	for (int i = e.size()-1; i >= 0; --i) {
		if (e[i].size()==0) continue;
		submenuIds.push_back(createSubmenu(e[i]));
	}

	/* Create the menu now, and populate it */
	int id = glutCreateMenu(GLUTWrapper::menuFunc);
	for (unsigned i = 0; i < e.size(); ++i) {
		if (e[i].size()==0) {
			addMenuEntry(e[i].getName());
			continue;
		}
		glutAddSubMenu(e[i].getName().c_str(), submenuIds.back());
		submenuIds.pop_back();
	}
	return id;
}

void GLUTWrapper::addMenuEntry(const std::string &name) {
	static unsigned menuId = 0;
	unsigned id = menuId++;
	menuEntries[id] = std::string(name.begin(), name.end());
	glutAddMenuEntry (name.c_str(), id);
}

void GLUTWrapper::createMenu(GLUTMenuEntry &menu)
{
	assert (menu.size() > 0);
	assert (menuEntries.size() == 0);
	createSubmenu(menu);
	glutAttachMenu (GLUT_RIGHT_BUTTON);
}

void GLUTWrapper::menuFunc(int id)
{
	assert (wrapper != NULL && wrapper->callbacks != NULL);
	wrapper->callbacks->menu(wrapper->menuEntries[id]);
}
