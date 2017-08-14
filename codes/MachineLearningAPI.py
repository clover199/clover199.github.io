from StatModels import *

plot_method = ["","png","pdf"][0]

if plot_method!="":
    import matplotlib
    matplotlib.use("cairo")
import matplotlib.pyplot as plt

from matplotlib import animation


def accuracy(pred, label):
    """
        returns the percentage of correct classification
    """
    if len(pred.shape)!=1:
        raise ValueError("input predicted values should be an 1d array")
    if len(label.shape)!=1:
        raise ValueError("input true values should be an 1d array")
    pred.astype(int)
    label.astype(int)
    return 1.0*np.sum(pred==label)/pred.shape[0]

def cross_entropy(pred, label):
    if len(pred.shape)!=2:
        raise ValueError("input predicted values should be a 2d array")
    if len(label.shape)!=1:
        raise ValueError("input true values should be an 1d array")
    y = np.zeros(pred.shape, dtype=bool)
    n = pred.shape[0]
    y[np.arange(n), label] = True
    return -np.mean(np.sum(np.where(y, np.log(pred), 0), axis=1))

def root_mean_square(pred, para):
    return np.sqrt(np.mean((pred-para)**2))

def soft_max(x):
    sm = np.exp(x-np.max(x,axis=1,keepdims=True))
    sm = sm/np.sum(sm,axis=1,keepdims=True)
    return sm


class data_base(object):
    """
        the basic class for explore data
        feature_shape must be of form (a,b)
    """
    def __init__(self, data=None, label=None, para=None, feature_shape=None):
        self._data = data
        self._label = label
        self._para = para
        self._feature_shape = feature_shape
        self._pca = None
        self._var = None

    def _plot_scatter(self, x, y):
        if self._para is not None:
            plt.scatter(x, y, c=self._para, s=40)
        elif self._label is not None:
            plt.scatter(x, y, c=self._label, s=40)
        else:
            plt.scatter(x, y, s=40)

    def _plot_imag(self, x):
        if x.ndim==2:
            plt.imshow(x, interpolation="nearest")
        else:
            if self._feature_shape is None:
                plt.plot(x, '-o')
            else:
                plt.imshow(x.reshape(self._feature_shape), interpolation="nearest")
        plt.colorbar()

    def append(self, data=None, label=None, para=None):
        if self._data is not None:
            if data is not None:
                self._data = np.concatenate((self._data,data),axis=0)
            else:
                raise ValueError("Please provide data")
        if self._label is not None:
            if label is not None:
                self._label = np.concatenate((self._label,label),axis=0)
            else:
                raise ValueError("Please provide label")
        if self._para is not None:
            if para is not None:
                self._para = np.concatenate((self._para,para),axis=0)
            else:
                raise ValueError("Please provide parameter")
        self._pca = None
        self._var = None

    def summary(self):
        print "\n********** summary of data **********"
        if self._data is None:
            print "No data"
            return 0
        if self._label is None:
            print "labels: not provided"
        else:
            print "labels: provided"
        if self._para is None:
            print "parameters: not provided"
        else:
            print "parameters: provided"

        print "sample and feature numbers:", self._data.shape
        print "min and max of data value:", np.min(self._data), np.max(self._data)

        if self._var is None:
            self._var = np.mean((self._data-np.mean(self._data, axis=0))**2, axis=0)
        print "number of zero-variance features:", sum(self._var<1e-16)
        print "min varance of non-zero variances:", min(self._var[self._var>1e-16])
        print "max varance of non-zero variances:", max(self._var), \
              "at", np.argmax(self._var)

        if self._pca is None:
            self._pca = PCA()
            self._pca.fit(self._data)
        comp = 0
        i = 0
        while comp<0.5:
            comp += self._pca.explained_variance_ratio_[i]
            i += 1
        print "first", i, "PCs explain 0.50 variance."
        while comp<0.75:
            comp += self._pca.explained_variance_ratio_[i]
            i += 1
        print "first", i, "PCs explain 0.75 variance."
        while comp<0.95:
            comp += self._pca.explained_variance_ratio_[i]
            i += 1
        print "first", i, "PCs explain 0.95 variance."
        while comp<0.99:
            comp += self._pca.explained_variance_ratio_[i]
            i += 1
        print "first", i, "PCs explain 0.99 variance."
        print "********** end of summary **********\n"

    def pca_plot(self, plot_type=None):
        """
            plot_type:  0:      plot variance vs PCs
                        i>0:    plot i-PC vs (i+1)-PC
                        -3<i<0: plot weight of features in |i|-PC
                        others: plot all
        """
        if self._data is None:
            print "No data"
            return 0

        if self._pca is None:
            self._pca = PCA()
            self._pca.fit(self._data)

        fitted = self._pca.transform(self._data)
        fig = plt.figure("pca_plot")

        if plot_type==-2:
            self._plot_imag(np.abs(self._pca.components_[1,:]))
            plt.title("Weight of features in 2nd PC")
        elif plot_type==-1:
            self._plot_imag(np.abs(self._pca.components_[0,:]))
            plt.title("Weight of features in 1st PC")
        elif plot_type==0:
            plt.plot(self._pca.explained_variance_[:10], '-o')
            plt.xlabel("Principal Pomponents")
            plt.ylabel("Variance")
        elif plot_type>0:
            self._plot_scatter(fitted[:,plot_type], fitted[:,plot_type-1])
            plt.xlabel("PC %d" % plot_type+1)
            plt.ylabel("PC %d" % plot_type)
        else:
            plt.subplot(2,2,1)
            self._plot_scatter(fitted[:,1], fitted[:,0])
            plt.ylabel("1st PC component")
            plt.subplot(2,2,2)
            plt.plot(self._pca.explained_variance_[:20], '-o')
            plt.subplot(2,2,3)
            self._plot_scatter(fitted[:,1], fitted[:,2])
            plt.xlabel("2nd PC component")
            plt.ylabel("3rd PC component")
            plt.subplot(2,2,4)
            self._plot_imag(np.abs(self._pca.components_[0,:]))

        if plot_method!="":
            plt.show()
        else:
            print "--> PCA plot saved as pca."+plot_method
            fig.savefig("pca."+plot_method)

    def video(self,):
        fig = plt.figure("video")
        ax1 = fig.add_subplot(1,2,1)

        if self._feature_shape is None:
            cor, = ax1.plot([],[], '-o')
            plt.xlim([0, self._data.shape[1]+0.1])
            plt.ylim([np.min(self._data), np.max(self._data)])
        else:
            im = ax1.imshow(self._data[0,:].reshape(self._feature_shape), \
                interpolation="nearest", vmin=np.min(self._data), vmax=np.max(self._data))
        # im = plt.imshow(data_x[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

        labels = np.zeros(1)
        if self._label is not None:
            labels = np.unique(self._label)
        para_min = 0
        para_max = 0
        if self._para is not None:
            para_min = np.min(self._para)
            para_max = np.max(self._para)

        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(np.array([para_min-0.1,para_max+0.1]), \
                np.tile(labels, (2,1)), 'k--')
        ax2.set_xlim([para_min-0.1,para_max+0.1])
        ax2.set_ylim([min(labels)-1, max(labels)+1])
        line, = ax2.plot([], [], 'ro')

        # function to update figure
        def updatefig(j):
            label = 0
            if self._label is not None:
                label = self._label[j]
            para = 0
            if self._para is not None:
                para = self._para[j]
            line.set_data([para], [label])

            if self._feature_shape is None:
                cor.set_data(np.arange(self._data.shape[1]), self._data[j,:])
                return cor,
            else:
                im.set_array(self._data[j,:].reshape(self._feature_shape))
                return im

        # kick off the animation
        ani = animation.FuncAnimation(fig, updatefig, frames=range(self._data.shape[0]),
                                      interval=500, blit=True)
        print "Animation saved as animation.mp4"
        ani.save('animation.mp4')

class data_disorder(data_base):
    
    def select_nonzero(self, plot_chosen=False):
        if self._var is None:
            self._var = np.mean((self._data-np.mean(self._data, axis=0))**2, axis=0)

        if plot_chosen:
            fig = plt.figure("select_nonzero")
            var = np.zeros(self._var.shape)
            var[self._var>1e-16] = 1
            self._plot_imag(var)
            if plot_method!="":
                plt.show()
            else:
                print "--> Non-zero-variance features saved as view."+plot_method
                fig.savefig("view."+plot_method)

        self._data = self._data[:,self._var>1e-16]
        self._feature_shape = None
        self._pca = None
        self._var = None

    def select_CC(self):
        """
            choose r(k)=<CC> with Im(r) with k even and Re(r) with k odd
            Im(BA0)   Re(AB1)   Im(AB2)
            Re(BA1)
            Im(BA2)
        """
        if self._feature_shape is not None:
            fs = np.array(self._feature_shape)
            index = np.arange(np.prod(fs)).reshape(fs)
            temp = np.zeros(fs//2, dtype=int)
            n = min(temp.shape)
            temp[np.arange(n), np.arange(n)] = np.diag(index,-1)[0::2]
            for i in range(1,n,2):
                # Re(AB)
                temp[np.arange(n-i), np.arange(n-i)+i] = np.diag(index,2*i+1)[::2]
                # Re(BA)
                temp[np.arange(n-i)+i, np.arange(n-i)] = np.diag(index,2*i-1)[1::2]
            for i in range(2,n,2):
                # Im(AB)
                temp[np.arange(n-i), np.arange(n-i)+i] = np.diag(index,-2*i+1)[1::2]
                # Im(BA)
                temp[np.arange(n-i)+i, np.arange(n-i)] = np.diag(index,-2*i-1)[::2]
            self._data = self._data[:, temp.flatten()]
            print("successfully selected non-zero-variance correlation")
            self._feature_shape = fs//2
            self._pca = None
            self._var = None
        else:
            print("please provide feature shape")

    def average_CC(self):
        """
            average r(k)=<CC> over k with Im(r) with k even and Re(r) with k odd
            features aranged as:
                Im(BA0) Re(AB1) Re(BA1) Im(AB2) Im(BA2) ...
        """
        if self._feature_shape is not None:
            fs = np.array(self._feature_shape)
            index = np.arange(np.prod(fs)).reshape(fs)
            temp = np.zeros(fs//2)
            n = min(temp.shape)
            new_data = np.mean(self._data[:,np.diag(index,-1)[0::2]], axis=1)
            for i in range(1,n,2):
                temp = np.mean(self._data[:, np.diag(index,2*i+1)[::2]], axis=1)
                new_data = np.concatenate((new_data, temp))
                temp = np.mean(self._data[:, np.diag(index,2*i-1)[1::2]], axis=1)
                new_data = np.concatenate((new_data, temp))
            for i in range(2,n,2):
                temp = np.mean(self._data[:, np.diag(index,-2*i+1)[1::2]], axis=1)
                new_data = np.concatenate((new_data, temp))
                temp = np.mean(self._data[:, np.diag(index,-2*i-1)[::2]], axis=1)
                new_data = np.concatenate((new_data, temp))
            self._data = new_data.reshape((2*n-1,-1)).transpose()
            print("successfully averaged non-zero-variance correlation")
            self._feature_shape = None
            self._pca = None
            self._var = None
        else:
            print("please provide feature shape")


class fit_base(object):
    """
        the basic class for fitting data with categorical response
    """
    def __init__(self, train=None, test=None, method="simple"):
        self._train = train
        self._test = test
        self._method = method

    def pca(self, keep=2, plotit=False):
        if self._train._pca is None:
            self._train._pca = PCA(n_components=keep)
            self._train._pca.fit(self._train._data)

        pca_fit = self._train._pca
        self._train._data = pca_fit.transform(self._train._data)
        self._test._data = pca_fit.transform(self._test._data)
        print("Choose %d PCs" % keep)

        if plotit:
            fig = plt.figure("pca")
            self._train._plot_scatter(self._train._data[:,1], self._train._data[:,0])
            self._test._plot_scatter(self._test._data[:,1], self._test._data[:,0])
            plt.xlabel("2nd PC component")
            plt.ylabel("1st PC component")
            if plot_method!="":
                plt.show()
            else:
                print "--> PCA plot saved as pca."+plot_method
                fig.savefig("pca."+plot_method)

class fit_cat(fit_base):
    """
        the basic class for fitting data with categorical response
    """
    def _fit(self):
        if self._method.lower()=="simple":
            print "\n********** training simple model: Logistic Regression **********"
            from sklearn.linear_model import LogisticRegression as lm
            self._simple_fit = lm()
            self._simple_fit.fit(self._train._data, self._train._label)
            train_label = self._simple_fit.predict(self._train._data)
            train_proba = self._simple_fit.predict_proba(self._train._data)
            test_label = self._simple_fit.predict(self._test._data)
            test_proba = self._simple_fit.predict_proba(self._test._data)
        elif self._method.lower()=="kmeans":
            print "\n********** training clustering model: K-Means **********"
            from sklearn.cluster import KMeans as km
            self._kmeans_fit = km(n_clusters=2)
            self._kmeans_fit.fit(self._train._data, self._train._label)
            train_label = self._kmeans_fit.predict(self._train._data)
            train_dist = self._kmeans_fit.transform(self._train._data)
            train_proba = soft_max(train_dist)
            test_label = self._kmeans_fit.predict(self._test._data)
            test_dist = self._kmeans_fit.transform(self._test._data)
            test_proba = soft_max(test_dist)
        else:
            raise ValueError("Method not found. Existing methods are \n" + \
                "\t Simple \n" + \
                "\t KMeans \n")

        if self._train._label is not None:
            print ("training accuracy and corss entropy: %.6f  %.6f" % \
                (accuracy(train_label, self._train._label), \
                cross_entropy(train_proba, self._train._label)))
        if self._test._label is not None:
            print ("testing accuracy and corss entropy:  %.6f  %.6f" % \
                (accuracy(test_label, self._test._label), \
                cross_entropy(test_proba, self._test._label)))
        print "********** end of training **********\n"

    def plot(self, data=None, label=0):
        if self._method.lower()=="simple":
            try:
                model_fit = self._simple_fit
            except AttributeError:
                self._fit()
                model_fit = self._simple_fit
            train_proba = model_fit.predict_proba(self._train._data)[:,label]
            test_proba = model_fit.predict_proba(self._test._data)[:,label]
        if self._method.lower()=="kmeans":
            try:
                model_fit = self._kmeans_fit
            except AttributeError:
                self._fit()
                model_fit = self._kmeans_fit
            train_proba = model_fit.transform(self._train._data)
            train_proba = soft_max(train_proba)[:,label]
            test_proba = model_fit.transform(self._test._data)
            test_proba = soft_max(test_proba)[:,label]

        fig = plt.figure("transition")
        if data=="train":
            if self._train._para is None:
                raise ValueError("Training data has no parameter")
            plt.plot(self._train._para, train_proba, 'bo')
            if self._train._label is not None:
                plt.plot(self._train._para, self._train._label, 'r-')
            plt.xlabel("Parameter")
            plt.ylabel("Probability")
        elif data=="test":
            if self._test._para is None:
                raise ValueError("Testing data has no parameter")
            plt.plot(self._test._para, test_proba, 'bo')
            plt.xlabel("Parameter")
            plt.ylabel("Probability")
        else:
            if self._train._para is None:
                raise ValueError("Training data has no parameter")
            if self._test._para is None:
                raise ValueError("Testing data has no parameter")
            plt.subplot(1,2,1)
            plt.plot(self._train._para, train_proba, 'bo')
            if self._train._label is not None:
                plt.plot(self._train._para, self._train._label, 'r-')
            plt.ylim([0,1])
            plt.xlabel("Parameter")
            plt.ylabel("Probability")
            plt.title("training data")
            plt.subplot(1,2,2)
            plt.plot(self._test._para, test_proba, 'bo')
            plt.ylim([0,1])
            plt.xlabel("Parameter")
            plt.title("testing data")
        if plot_method!="":
            plt.show()
        else:
            print "--> Transition saved as transition."+plot_method
            fig.savefig("transition."+plot_method)
        class fit_cat(object):
            """
                the basic class for fitting data with categorical response
            """
            def __init__(self, train=None, test=None, method="simple"):
                self._train = train
                self._test = test
                self._method = method

            def _fit(self):
                if self._method.lower()=="simple":
                    print "\n********** training simple model: Logistic Regression **********"
                    from sklearn.linear_model import LogisticRegression as lm
                    self._simple_fit = lm()
                    self._simple_fit.fit(self._train._data, self._train._label)
                    train_label = self._simple_fit.predict(self._train._data)
                    train_proba = self._simple_fit.predict_proba(self._train._data)
                    test_label = self._simple_fit.predict(self._test._data)
                    test_proba = self._simple_fit.predict_proba(self._test._data)
                elif self._method.lower()=="kmeans":
                    print "\n********** training clustering model: K-Means **********"
                    from sklearn.cluster import KMeans as km
                    self._kmeans_fit = km(n_clusters=2)
                    self._kmeans_fit.fit(self._train._data, self._train._label)
                    train_label = self._kmeans_fit.predict(self._train._data)
                    train_dist = self._kmeans_fit.transform(self._train._data)
                    train_proba = soft_max(train_dist)
                    test_label = self._kmeans_fit.predict(self._test._data)
                    test_dist = self._kmeans_fit.transform(self._test._data)
                    test_proba = soft_max(test_dist)
                else:
                    raise ValueError("Method not found. Existing methods are \n" + \
                        "\t Simple \n" + \
                        "\t KMeans \n")

                if self._train._label is not None:
                    print ("training accuracy and corss entropy: %.6f  %.6f" % \
                        (accuracy(train_label, self._train._label), \
                        cross_entropy(train_proba, self._train._label)))
                if self._test._label is not None:
                    print ("testing accuracy and corss entropy:  %.6f  %.6f" % \
                        (accuracy(test_label, self._test._label), \
                        cross_entropy(test_proba, self._test._label)))
                print "********** end of training **********\n"

            def pca(self, keep=2, plotit=False):
                if self._train._pca is None:
                    self._train._pca = PCA(n_components=keep)
                    self._train._pca.fit(self._train._data)

                pca_fit = self._train._pca
                self._train._data = pca_fit.transform(self._train._data)
                self._test._data = pca_fit.transform(self._test._data)
                print("Choose %d PCs" % keep)

                if plotit:
                    fig = plt.figure("pca")
                    self._train._plot_scatter(self._train._data[:,1], self._train._data[:,0])
                    self._test._plot_scatter(self._test._data[:,1], self._test._data[:,0])
                    plt.xlabel("2nd PC component")
                    plt.ylabel("1st PC component")
                    if plot_method!="":
                        plt.show()
                    else:
                        print "--> PCA plot saved as pca."+plot_method
                        fig.savefig("pca."+plot_method)

            def plot(self, data=None, label=0):
                if self._method.lower()=="simple":
                    try:
                        model_fit = self._simple_fit
                    except AttributeError:
                        self._fit()
                        model_fit = self._simple_fit
                    train_proba = model_fit.predict_proba(self._train._data)[:,label]
                    test_proba = model_fit.predict_proba(self._test._data)[:,label]
                if self._method.lower()=="kmeans":
                    try:
                        model_fit = self._kmeans_fit
                    except AttributeError:
                        self._fit()
                        model_fit = self._kmeans_fit
                    train_proba = model_fit.transform(self._train._data)
                    train_proba = soft_max(train_proba)[:,label]
                    test_proba = model_fit.transform(self._test._data)
                    test_proba = soft_max(test_proba)[:,label]

                fig = plt.figure("transition")
                if data=="train":
                    if self._train._para is None:
                        raise ValueError("Training data has no parameter")
                    plt.plot(self._train._para, train_proba, 'bo')
                    if self._train._label is not None:
                        plt.plot(self._train._para, self._train._label, 'r-')
                    plt.xlabel("Parameter")
                    plt.ylabel("Probability")
                elif data=="test":
                    if self._test._para is None:
                        raise ValueError("Testing data has no parameter")
                    plt.plot(self._test._para, test_proba, 'bo')
                    plt.xlabel("Parameter")
                    plt.ylabel("Probability")
                else:
                    if self._train._para is None:
                        raise ValueError("Training data has no parameter")
                    if self._test._para is None:
                        raise ValueError("Testing data has no parameter")
                    plt.subplot(1,2,1)
                    plt.plot(self._train._para, train_proba, 'bo')
                    if self._train._label is not None:
                        plt.plot(self._train._para, self._train._label, 'r-')
                    plt.ylim([0,1])
                    plt.xlabel("Parameter")
                    plt.ylabel("Probability")
                    plt.title("training data")
                    plt.subplot(1,2,2)
                    plt.plot(self._test._para, test_proba, 'bo')
                    plt.ylim([0,1])
                    plt.xlabel("Parameter")
                    plt.title("testing data")
                if plot_method!="":
                    plt.show()
                else:
                    print "--> Transition saved as transition."+plot_method
                    fig.savefig("transition."+plot_method)

class fit_num(fit_base):
    """
        the basic class for fitting data with numerical response
    """
    def _fit(self):
        if self._method.lower()=="simple":
            print "\n********** training simple model: Simple Linear Regression **********"
            from sklearn.linear_model import LinearRegression as lm
            self._simple_fit = lm()
            self._simple_fit.fit(self._train._data, self._train._para)
            train_para = self._simple_fit.predict(self._train._data)
            test_para = self._simple_fit.predict(self._test._data)

        if self._train._para is not None:
            print ("training error: %.6f" % \
                (root_mean_square(train_para, self._train._para)))
        if self._test._para is not None:
            print ("testing error:  %.6f" % \
                (root_mean_square(test_para, self._test._para)))
        print "********** end of training **********\n"

if __name__ == "__main__":
    data = data_base()
