#include <utility>

/** Checks if a given function throws for given arguments.
 * @param f function
 * @param args set of arguments
 */
template<class F, class... Args>
bool doesThrow(F&& f, Args&&... args)
{
  try {
    f(std::forward<Args>(args)...);
  }
  catch(...) { return true; }

  return false;
}
